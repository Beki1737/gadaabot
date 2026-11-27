"""Post hoc robustness experiments for the GadaaBot paper (council round 1, decisions D60/D61).

All rules below were fixed before running. The 184 reviewed test questions and the blind sheet are used as
test data only (labelled post hoc in the paper); thresholds are calibrated on the 132 held-out items that the
native-speaker review dropped, never on the test set.

Subcommands
  judge-inputs  E1  which inputs make an LLM judge agree with humans? variants:
                    ref (question + reference), ctx (question + passages), ref_ctx (both),
                    ref_ctx_nocite (both, inline [n] markers stripped from the answer)
                    on the blind sheet (150 answers, 2 annotators) and the dev set (108 answers, 1 annotator);
                    plus the citation-marker control on all 184 test answers of the 8 base systems.
                    --judge-url/--judge-model let a second judge family run the same protocol (E1b).
  abstain       E4  reranker-score abstention for single-pass native RAG (CPU, existing logs)
  tierfilter    E5  native RAG restricted to redistributable/permitted tiers (A + P), the legal public index
  seeds         E6  extra sampling seeds for closed, rag_naive, agent_noverify, agent_full

Pre-registered expectations (what would disprove them) are printed with each result block.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gadaabot.agent.graph import answer, build_graph
from gadaabot.agent.llm import ANSWER, LLM
from gadaabot.agent.retriever import COLL, PRIOR, Evidence, Retriever
from gadaabot.eval.ablation import CLOSED_SYS, GRAPHS, LockedRetriever, boot_p, holm
from gadaabot.eval.judge import Judge, answer_body, boot_ci, qwk, spearman

BASE_ARMS = ["closed", "closed_tt_nllb", "closed_tt_self", "rag_naive", "rag_tt_nllb", "rag_tt_self",
             "agent_noverify", "agent_full"]
CITE = re.compile(r"\s*\[\d+\](?:\s*\[\d+\])*")
TOP = re.compile(r"retrieve #1:(.*)")


# ------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, root: Path) -> None:
        self.root = root
        dv = root / "data/eval/gadaabot_dev"
        self.dev = {d["id"]: d for d in map(json.loads, (dv / "devset_candidates.jsonl").read_text(encoding="utf-8").splitlines()) if d}
        rev = list(csv.DictReader((dv / "devset_review.tsv").open(encoding="utf-8"), delimiter="\t"))
        self.keep = {r["id"] for r in rev if r["keep_y_n"].strip().lower().startswith("y")}
        self.iter_ids = {json.loads(l)["id"] for l in (root / "runs/T027_agent/answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        held = set(self.dev) - self.iter_ids
        self.test = sorted(held & self.keep)
        self.calib = sorted(held - self.keep)  # review-dropped held-out items: calibration only
        self.v5 = root / "runs/T030_ablation"
        self.v4 = root / "runs/T030_ablation_v4"

    def arm(self, name: str, v4: bool = False) -> dict[str, dict[str, Any]]:
        f = (self.v4 if v4 else self.v5) / f"arm_{name}.jsonl"
        return {r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r}

    def judged(self, v4: bool = False) -> dict[tuple[str, str], float]:
        out = {}
        for l in ((self.v4 if v4 else self.v5) / "judge.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("mean") is not None:
                    out[(r["id"], r["arm"])] = float(r["mean"])
        return out


def mean_ci(v: list[float]) -> tuple[float, float, float]:
    lo, hi = boot_ci(lambda s: statistics.fmean(v[t] for t in s), len(v))
    return round(statistics.fmean(v), 3), round(lo, 3), round(hi, 3)


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    ids = [i for i in a if i in b]
    d = [a[i] - b[i] for i in ids]
    m, lo, hi = mean_ci(d)
    return {"n": len(ids), "diff": m, "lo": lo, "hi": hi, "p": round(boot_p(d), 4)}


def kendall(x: list[float], y: list[float]) -> float:
    c = dsc = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = (x[i] - x[j]) * (y[i] - y[j])
            c += s > 0
            dsc += s < 0
    return (c - dsc) / max(1, c + dsc)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", restval="")
            w.writeheader()
            w.writerows(rows)


def make_retriever(args: argparse.Namespace, cls: type | None = None, **kw: Any) -> Retriever:  # noqa: ANN401
    return (cls or Retriever)(args.qdrant_url, args.afrie5, args.reranker, args.parents, **kw)


def backgrounds(ret: Any, items: list[dict[str, Any]]) -> dict[str, list[str]]:  # noqa: ANN401
    """Judge background = top-3 parents retrieved for the clean question (same protocol as T030/T031)."""
    return {d["id"]: [e.text for e in ret.search(d["question"], k_parents=4)[:3]] for d in items}


# ------------------------------------------------------------------------------------------- E1
VARIANTS = ("ref", "ctx", "ref_ctx", "ref_ctx_nocite")


def job(variant: str, q: str, ans: str, ref: str, psg: list[str]) -> dict[str, Any]:
    if variant == "ref":
        return {"variant": "ref", "question": q, "answer": ans, "reference": ref, "passages": []}
    if variant == "ctx":
        return {"variant": "ref_ctx", "question": q, "answer": ans, "reference": "", "passages": psg}
    if variant == "ref_ctx_nocite":
        return {"variant": "ref_ctx", "question": q, "answer": CITE.sub("", answer_body(ans)), "reference": ref, "passages": psg}
    return {"variant": "ref_ctx", "question": q, "answer": ans, "reference": ref, "passages": psg}


def agreement(rows: list[tuple[str, str, float, float]], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """rows: (qid, system, human, judge_mean). Item agreement + system-level agreement."""
    h = [int(round(r[2])) for r in rows]
    j = [int(round(r[3])) for r in rows]
    lo, hi = boot_ci(lambda s: qwk([j[t] for t in s], [h[t] for t in s]), len(rows))
    sysm_h: dict[str, list[float]] = defaultdict(list)
    sysm_j: dict[str, list[float]] = defaultdict(list)
    for _, s, hv, jv in rows:
        sysm_h[s].append(hv)
        sysm_j[s].append(jv)
    systems = sorted(sysm_h)
    mh = [statistics.fmean(sysm_h[s]) for s in systems]
    mj = [statistics.fmean(sysm_j[s]) for s in systems]
    agree = tot = 0
    for i in range(len(systems)):
        for k in range(i + 1, len(systems)):
            tot += 1
            agree += (mh[i] - mh[k]) * (mj[i] - mj[k]) > 0
    out: dict[str, Any] = {"n": len(rows), "qwk": round(qwk(j, h), 3), "qwk_lo": round(lo, 3), "qwk_hi": round(hi, 3),
                           "spearman": round(spearman([r[3] for r in rows], [r[2] for r in rows]), 3),
                           "bias": round(statistics.fmean(r[3] - r[2] for r in rows), 3),
                           "kendall_systems": round(kendall(mh, mj), 3), "pair_sign_agree": f"{agree}/{tot}"}
    by = defaultdict(dict)
    for q, s, hv, jv in rows:
        by[s][q] = (hv, jv)
    for a, b in pairs:
        common = [q for q in by[a] if q in by[b]]
        dh = [by[a][q][0] - by[b][q][0] for q in common]
        dj = [by[a][q][1] - by[b][q][1] for q in common]
        out[f"{a}-{b}"] = f"human {statistics.fmean(dh):+.2f} judge {statistics.fmean(dj):+.2f} " \
                          f"[{boot_ci(lambda s, dj=dj: statistics.fmean(dj[t] for t in s), len(dj))[0]:+.2f}, " \
                          f"{boot_ci(lambda s, dj=dj: statistics.fmean(dj[t] for t in s), len(dj))[1]:+.2f}]"
    return out


def cmd_judge_inputs(args: argparse.Namespace) -> None:
    D = Data(args.root)
    ret = LockedRetriever(make_retriever(args))
    jl = LLM(args.judge_url, args.judge_model)
    judge = Judge(jl, args.samples, workers=16)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    T = args.root / "reports/tables"

    # blind sheet: v4 answers that the humans scored
    key = json.loads((D.v4 / "blind_key_v2.json").read_text(encoding="utf-8"))
    a1 = {r["row"]: float(r["correct_1to5"]) for r in csv.DictReader((T / "T031_blind_human_v2.tsv").open(encoding="utf-8"), delimiter="\t")}
    a2 = {r["row"]: float(r["correct_1to5"]) for r in csv.DictReader((T / "T032_blind_annotator2.tsv").open(encoding="utf-8"), delimiter="\t")}
    arms4 = {a: D.arm(a, v4=True) for a in {v["arm"] for v in key.values()}}
    blind = [(row, key[row]["id"], key[row]["arm"], arms4[key[row]["arm"]][key[row]["id"]]["answer"], (a1[row] + a2[row]) / 2)
             for row in sorted(key, key=int) if row in a1 and row in a2]
    # dev set (T027): one annotator
    t27 = {r["id"]: r for r in csv.DictReader((T / "T027_rag_vs_closed.tsv").open(encoding="utf-8"), delimiter="\t")}
    r27 = {r["id"]: r for r in map(json.loads, (args.root / "runs/T027_agent/answers.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    dev = []
    for i, r in t27.items():
        if i in r27 and r["rag_correct_1to5"].strip() and r["closed_correct_1to5"].strip():
            dev.append((i, "rag", str(r27[i].get("rag_answer", "")), float(r["rag_correct_1to5"]), r27[i]))
            dev.append((i, "closed", str(r27[i].get("closed_book", "")), float(r["closed_correct_1to5"]), r27[i]))
    qinfo = {d["id"]: d for d in D.dev.values()}
    for i, _, _, _, rr in dev:
        qinfo.setdefault(i, {"id": i, "question": rr["question"], "answer": rr.get("gold", "")})
    bg = backgrounds(ret, [qinfo[i] for i in sorted({b[1] for b in blind} | {d[0] for d in dev})])
    rows_out = []
    print("E1 pre-registered: passage-grounded variants (ctx, ref_ctx) beat ref on system-level agreement on BOTH sets; "
          "disproved if ref has equal or better pair_sign_agree and kendall on the blind sheet.", flush=True)
    for v in VARIANTS:
        judge = Judge(jl, args.samples, workers=16)  # one cache per variant (ctx and ref_ctx share the prompt variant name)
        jobs = [job(v, qinfo[qid]["question"], ans, qinfo[qid].get("answer", ""), bg[qid]) for _, qid, _, ans, _ in blind]
        jobs += [job(v, qinfo[qid]["question"], ans, qinfo[qid].get("answer", ""), bg[qid]) for qid, _, ans, _, _ in dev]
        sc = judge.score_many(jobs)
        nb = len(blind)
        rb = [(qid, s, hv, x["mean"]) for (_, qid, s, _, hv), x in zip(blind, sc[:nb]) if x["mean"] is not None]
        rd = [(qid, s, hv, x["mean"]) for (qid, s, _, hv, _), x in zip(dev, sc[nb:]) if x["mean"] is not None]
        for name, rws, pairs in (("blind", rb, [("rag_naive", "closed"), ("rag_naive", "rag_tt_nllb"), ("agent_full", "rag_naive"),
                                                ("closed_tt_self", "closed")]),
                                 ("dev", rd, [("rag", "closed")])):
            m = agreement(rws, pairs)
            rows_out.append({"judge": args.tag, "set": name, "variant": v, **m})
            print(f"E1 [{args.tag}] {name:<5} {v:<15} " + " | ".join(f"{k} {val}" for k, val in m.items()), flush=True)
    write_tsv(out / f"E1_{args.tag}_agreement.tsv", rows_out)

    # citation-marker control on the 184 test answers (v5 systems)
    if args.test_control:
        judged = D.judged()  # ref_ctx scores of the 184 test answers (T030 to T032)
        items = [D.dev[i] for i in D.test]
        bgt = backgrounds(ret, items)
        res = {}
        judge = Judge(jl, args.samples, workers=16)
        for a in BASE_ARMS:
            arm = D.arm(a)
            jobs = [job("ref_ctx_nocite", d["question"], arm[d["id"]]["answer"], d["answer"], bgt[d["id"]]) for d in items]
            res[a] = {d["id"]: x["mean"] for d, x in zip(items, judge.score_many(jobs)) if x["mean"] is not None}
        rows = []
        for a in BASE_ARMS:
            with_c = [judged[(i, a)] for i in res[a] if (i, a) in judged]
            rows.append({"arm": a, "ref_ctx": round(statistics.fmean(with_c), 3), "ref_ctx_nocite": round(statistics.fmean(res[a].values()), 3),
                         "shift": round(statistics.fmean(res[a].values()) - statistics.fmean(with_c), 3),
                         "has_markers_pct": round(100 * sum(bool(CITE.search(answer_body(D.arm(a)[i]["answer"]))) for i in res[a]) / len(res[a]), 1)})
            print("E1c " + " | ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
        pr = []
        for a, b in (("rag_naive", "rag_tt_self"), ("rag_naive", "rag_tt_nllb"), ("rag_naive", "closed")):
            pr.append({"a": a, "b": b, **paired(res[a], res[b])})
        for p, ph in zip(pr, holm([p["p"] for p in pr])):
            p["p_holm"] = round(ph, 4)
            print(f"E1c-PAIR nocite {p['a']} - {p['b']}: {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] p_holm={p['p_holm']}", flush=True)
        print("E1c pre-registered: citation bonus exists if rag_naive shift < -0.15 while cascade shifts are within 0.05; "
              "the native-vs-cascade claim survives if the nocite pairs keep CIs above 0.", flush=True)
        write_tsv(out / f"E1c_{args.tag}_nocite.tsv", rows + pr)


# ------------------------------------------------------------------------------------------- E4
def top_score(trace: list[str]) -> float | None:
    for t in trace or []:
        m = TOP.match(t)
        if m:
            v = [float(x) for x in re.findall(r"[+-]\d+(?:\.\d+)?", m.group(1))]
            return max(v) if v else None
    return None


def cmd_abstain(args: argparse.Namespace) -> None:
    D = Data(args.root)
    J = D.judged()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def collect(arm: str, ids: list[str]) -> list[tuple[str, float, float]]:
        a = D.arm(arm)
        return [(i, top_score(a[i].get("trace", [])), J[(i, arm)]) for i in ids
                if i in a and (i, arm) in J and top_score(a[i].get("trace", [])) is not None]

    cal = collect("rag_naive", D.calib) + collect("rag_naive@nogold", D.calib)
    best = (0.0, None)
    for tau in sorted({c[1] for c in cal}):
        tp = sum(1 for _, s, j in cal if s < tau and j <= 2)
        fn = sum(1 for _, s, j in cal if s >= tau and j <= 2)
        tn = sum(1 for _, s, j in cal if s >= tau and j > 2)
        fp = sum(1 for _, s, j in cal if s < tau and j > 2)
        bacc = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if bacc > best[0]:
            best = (bacc, tau)
    tau = best[1]
    print(f"E4 calibration on {len(cal)} review-dropped held-out answers: tau = {tau} (balanced accuracy {best[0]:.3f})", flush=True)
    print("E4 pre-registered: reranker-gated RAG has fewer low-scored answers than agent_noverify and agent_full under gold "
          "removal at no lower answered quality; disproved if its low-scored rate is not below both.", flush=True)
    rows = []
    for cond in ("", "@nogold"):
        rag = collect("rag_naive" + cond, D.test)
        ans = [r for r in rag if r[1] >= tau]
        row = {"system": "rag_naive+rerank_gate" + cond, "n": len(rag), "coverage_pct": round(100 * len(ans) / len(rag), 1),
               "answered_mean": round(statistics.fmean(r[2] for r in ans), 3) if ans else "",
               "low_scored_pct_of_all": round(100 * sum(r[2] <= 2 for r in ans) / len(rag), 1),
               "abstain_pct": round(100 * (len(rag) - len(ans)) / len(rag), 1)}
        rows.append(row)
        print("E4 " + " | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        base = {"system": "rag_naive (no gate)" + cond, "n": len(rag), "coverage_pct": 100.0,
                "answered_mean": round(statistics.fmean(r[2] for r in rag), 3),
                "low_scored_pct_of_all": round(100 * sum(r[2] <= 2 for r in rag) / len(rag), 1), "abstain_pct": 0.0}
        rows.append(base)
        print("E4 " + " | ".join(f"{k} {v}" for k, v in base.items()), flush=True)
        for ag in ("agent_noverify", "agent_full"):
            a = D.arm(ag + cond)
            ids = [i for i in D.test if i in a and (i, ag + cond) in J]
            answered = [i for i in ids if not a[i].get("abstained")]
            r = {"system": ag + cond, "n": len(ids), "coverage_pct": round(100 * len(answered) / len(ids), 1),
                 "answered_mean": round(statistics.fmean(J[(i, ag + cond)] for i in answered), 3),
                 "low_scored_pct_of_all": round(100 * sum(J[(i, ag + cond)] <= 2 for i in answered) / len(ids), 1),
                 "abstain_pct": round(100 * (len(ids) - len(answered)) / len(ids), 1)}
            rows.append(r)
            print("E4 " + " | ".join(f"{k} {v}" for k, v in r.items()), flush=True)
            # descriptive matched-coverage comparison (post hoc): rag gated to the agent's coverage
            k = len(answered)
            ranked = sorted(rag, key=lambda x: -x[1])[:k]
            print(f"E4-MATCHED{cond or '@clean'} rag gated to {ag} coverage {r['coverage_pct']}%: answered_mean "
                  f"{statistics.fmean(x[2] for x in ranked):.3f} low_scored_pct_of_all {100 * sum(x[2] <= 2 for x in ranked) / len(rag):.1f} "
                  f"vs {ag} {r['answered_mean']} / {r['low_scored_pct_of_all']}", flush=True)
        # risk-coverage area (lower is better): reranker ordering vs random expectation
        srt = sorted(rag, key=lambda x: -x[1])
        risk = [statistics.fmean(x[2] <= 2 for x in srt[:k]) for k in range(1, len(srt) + 1)]
        base_risk = statistics.fmean(x[2] <= 2 for x in rag)
        print(f"E4-AURC{cond or '@clean'} reranker {statistics.fmean(risk):.3f} vs no-signal {base_risk:.3f}", flush=True)
    write_tsv(out / "E4_abstention.tsv", rows)


# ------------------------------------------------------------------------------------------- E5
class TierRetriever(Retriever):
    """Retriever restricted to given license tiers (payload filter inside both prefetch legs)."""

    def __init__(self, *a: Any, tiers: tuple[str, ...] = ("A", "P"), **k: Any) -> None:  # noqa: ANN401
        super().__init__(*a, **k)
        self.tiers = list(tiers)

    def search(self, query: str, k_parents: int = 4, k_children: int = 50, authority_boost: float = 0.0) -> list[Evidence]:
        import torch
        from qdrant_client import models

        flt = models.Filter(must=[models.FieldCondition(key="tier", match=models.MatchAny(any=self.tiers))])
        idx, val = self.bm.encode_query(query)
        with torch.no_grad():
            qv = self._qvec(query)
        pts = self.qc.query_points(COLL, prefetch=[
            models.Prefetch(query=qv, using="afrie5", limit=100, filter=flt),
            models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="bm25_trunc4", limit=100, filter=flt)],
            query=models.FusionQuery(fusion=models.Fusion.DBSF), limit=k_children, with_payload=True).points
        if not pts:
            return []
        texts = [str(p.payload["text"]) for p in pts]
        with torch.no_grad():
            b = self.rtok([query] * len(texts), texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            scores = self.rr(**b).logits.view(-1).float().cpu().tolist()
        best: dict[str, tuple[float, str]] = {}
        for p, s in zip(pts, scores):
            pl = p.payload or {}
            s += authority_boost * PRIOR.get(str(pl.get("source")), 0.0)
            pid = str(pl["parent_id"])
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, str(pl["text"]))
        out = []
        for pid, (s, child) in sorted(best.items(), key=lambda x: -x[1][0])[:k_parents]:
            row = self.db.execute("SELECT source, tier, license, domain, title, url, text FROM parents WHERE parent_id=?", (pid,)).fetchone()
            if row:
                out.append(Evidence(pid, row[0], row[1], row[2], row[3], row[4], row[5], row[6], round(s, 3), child))
        return out


def cmd_tierfilter(args: argparse.Namespace) -> None:
    D = Data(args.root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.llm_url, args.llm_model)
    full = LockedRetriever(make_retriever(args))
    tier = LockedRetriever(make_retriever(args, TierRetriever, tiers=("A", "P")))
    items = [D.dev[i] for i in D.test]
    app = build_graph(llm, tier, **GRAPHS["rag_naive"])
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip([d["id"] for d in items], ex.map(lambda d: answer(app, d["question"]), items)))
    with (out / "E5_tierAP_answers.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in res.items():
            fh.write(json.dumps({"id": i, "answer": r["answer"], "abstained": r["abstained"], "evidence_ids": r["evidence_ids"]}, ensure_ascii=False) + "\n")
    bg = backgrounds(full, items)
    judge = Judge(llm, args.samples, workers=16)
    sc = judge.score_many([job("ref_ctx", d["question"], res[d["id"]]["answer"], d["answer"], bg[d["id"]]) for d in items])
    tierA = {i: x["mean"] for i, x in zip(res, sc) if x["mean"] is not None}
    J = D.judged()
    full_s = {i: J[(i, "rag_naive")] for i in D.test if (i, "rag_naive") in J}
    closed_s = {i: J[(i, "closed")] for i in D.test if (i, "closed") in J}
    db = sqlite3.connect(str(args.parents))
    gold_tier = {i: (db.execute("SELECT tier FROM parents WHERE parent_id=?", (D.dev[i]["parent_id"],)).fetchone() or ["?"])[0] for i in D.test}
    print("E5 pre-registered: none (descriptive). Expect a drop for questions whose gold passage is tier C (web, mdh).", flush=True)
    for name, ids in (("all", D.test), ("gold A/P", [i for i in D.test if gold_tier[i] in ("A", "P")]),
                      ("gold C", [i for i in D.test if gold_tier[i] == "C"])):
        a = {i: tierA[i] for i in ids if i in tierA}
        print(f"E5 {name:<9} n={len(a)} rag_tierAP {statistics.fmean(a.values()):.3f} | rag_full {statistics.fmean(full_s[i] for i in a):.3f} | "
              f"closed {statistics.fmean(closed_s[i] for i in a):.3f} | tierAP - full {paired(a, full_s)} | tierAP - closed {paired(a, closed_s)}", flush=True)


# ------------------------------------------------------------------------------------------- E6
class SeededLLM(LLM):
    def __init__(self, url: str, model: str, seed: int) -> None:
        super().__init__(url, model)
        self.seed = seed

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 700, sampling: dict[str, float] | None = None) -> str:
        return super().chat(messages, max_tokens=max_tokens, sampling={**(sampling or ANSWER), "seed": self.seed})


def cmd_seeds(args: argparse.Namespace) -> None:
    D = Data(args.root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ret = LockedRetriever(make_retriever(args))
    items = [D.dev[i] for i in D.test]
    arms = ["closed", "rag_naive", "agent_noverify", "agent_full"]
    J = D.judged()
    scores: dict[str, dict[str, dict[str, float]]] = {a: {"orig": {i: J[(i, a)] for i in D.test if (i, a) in J}} for a in arms}
    judge = Judge(LLM(args.llm_url, args.llm_model), args.samples, workers=16)
    bg = backgrounds(ret, items)
    for seed in args.seed_list:
        llm = SeededLLM(args.llm_url, args.llm_model, seed)
        for a in arms:
            f = out / f"E6_seed{seed}_{a}.jsonl"
            if f.exists():
                got = {r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r}
            else:
                if a == "closed":
                    def run(d: dict[str, Any]) -> dict[str, Any]:
                        return {"answer": llm.chat([{"role": "system", "content": CLOSED_SYS}, {"role": "user", "content": d["question"]}], max_tokens=500)}
                else:
                    app = build_graph(llm, ret, **GRAPHS[a])

                    def run(d: dict[str, Any], app: Any = app) -> dict[str, Any]:  # noqa: ANN401
                        r = answer(app, d["question"])
                        return {"answer": r["answer"], "abstained": r["abstained"]}
                with ThreadPoolExecutor(8) as ex:
                    got = dict(zip([d["id"] for d in items], ex.map(run, items)))
                f.write_text("\n".join(json.dumps({"id": i, **r}, ensure_ascii=False) for i, r in got.items()) + "\n", encoding="utf-8")
            sc = judge.score_many([job("ref_ctx", d["question"], got[d["id"]]["answer"], d["answer"], bg[d["id"]]) for d in items])
            scores[a][f"s{seed}"] = {d["id"]: x["mean"] for d, x in zip(items, sc) if x["mean"] is not None}
            print(f"E6 seed {seed} {a:<15} mean {statistics.fmean(scores[a][f's{seed}'].values()):.3f}", flush=True)
    runs = list(scores[arms[0]])
    print("E6 pre-registered: the ordering rag_naive >= agent_noverify > agent_full > closed holds in every run; "
          "rag_naive - agent_noverify pooled CI includes 0 or is positive.", flush=True)
    for r in runs:
        order = sorted(arms, key=lambda a: -statistics.fmean(scores[a][r].values()))
        print(f"E6 run {r:<6} order {' > '.join(order)} | " + " ".join(f"{a} {statistics.fmean(scores[a][r].values()):.3f}" for a in arms), flush=True)
    pooled = {a: {i: statistics.fmean(scores[a][r][i] for r in runs if i in scores[a][r]) for i in D.test
                  if all(i in scores[a][r] for r in runs)} for a in arms}
    sd = {a: statistics.fmean(statistics.pstdev([scores[a][r][i] for r in runs]) for i in pooled[a]) for a in arms}
    pr = [{"a": a, "b": b, **paired(pooled[a], pooled[b])} for a, b in
          (("rag_naive", "closed"), ("agent_noverify", "rag_naive"), ("agent_full", "rag_naive"), ("agent_full", "agent_noverify"))]
    for p, ph in zip(pr, holm([p["p"] for p in pr])):
        p["p_holm"] = round(ph, 4)
        print(f"E6-PAIR pooled over {len(runs)} runs {p['a']} - {p['b']}: {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] p_holm={p['p_holm']}", flush=True)
    print("E6 per-question SD across runs: " + " ".join(f"{a} {v:.2f}" for a, v in sd.items()), flush=True)
    write_tsv(out / "E6_pairs.tsv", pr)


# ------------------------------------------------------------------------------------------- E2 / E1b: re-judge
E2_PAIRS = [("rag_naive", "closed"), ("rag_naive", "rag_tt_self"), ("rag_naive", "rag_tt_nllb"), ("closed_tt_self", "closed"),
            ("agent_noverify", "rag_naive"), ("agent_full", "rag_naive"), ("rag_tt_self", "rag_tt_nllb")]


def cmd_rejudge(args: argparse.Namespace) -> None:
    """Judge saved arm files (any backbone) on the 184 test questions with the fixed ref_ctx protocol."""
    D = Data(args.root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ret = LockedRetriever(make_retriever(args))
    items = ([json.loads(l) for l in args.items_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
             if args.items_jsonl else [D.dev[i] for i in D.test])  # --items-jsonl: e.g. the real-question set
    bg = backgrounds(ret, items)
    judge = Judge(LLM(args.judge_url, args.judge_model), args.samples, workers=16)
    sc: dict[str, dict[str, float]] = {}
    with (out / f"rejudge_{args.tag}.jsonl").open("w", encoding="utf-8") as fh:
        for a in args.arm_list:
            f = args.arms_dir / f"arm_{a}.jsonl"
            if not f.exists():
                print(f"E2 [{args.tag}] skip {a}: no arm file", flush=True)
                continue
            arm = {r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r}
            its = [d for d in items if d["id"] in arm]
            res = judge.score_many([job("ref_ctx", d["question"], arm[d["id"]]["answer"], d["answer"], bg[d["id"]]) for d in its])
            sc[a] = {d["id"]: x["mean"] for d, x in zip(its, res) if x["mean"] is not None}
            for i, v in sc[a].items():
                fh.write(json.dumps({"id": i, "arm": a, "mean": v}) + "\n")
            m, lo, hi = mean_ci(list(sc[a].values()))
            ab = sum(bool(arm[i].get("abstained")) for i in sc[a])
            print(f"E2 [{args.tag}] {args.arms_dir.name} {a:<15} n={len(sc[a])} judge {m:.3f} [{lo:.2f}, {hi:.2f}] abstain {100 * ab / len(sc[a]):.1f}%", flush=True)
    pr = [{"a": a, "b": b, **paired(sc[a], sc[b])} for a, b in E2_PAIRS if a in sc and b in sc]
    for p, ph in zip(pr, holm([p["p"] for p in pr])):
        p["p_holm"] = round(ph, 4)
        print(f"E2-PAIR [{args.tag}] {args.arms_dir.name} {p['a']} - {p['b']}: {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] "
              f"n={p['n']} p_holm={p['p_holm']}", flush=True)
    write_tsv(out / f"rejudge_{args.tag}_{args.arms_dir.name}_pairs.tsv", pr)


# ------------------------------------------------------------------------------------------- E3: MT quality
def chrf(hyps: list[str], refs: list[str], n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF (character n-grams 1..6, beta 2, whitespace ignored), as in Popovic (2015)."""
    from collections import Counter
    match, tp, tr = [0] * n, [0] * n, [0] * n
    for h, r in zip(hyps, refs):
        h, r = h.replace(" ", ""), r.replace(" ", "")
        for k in range(1, n + 1):
            hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
            rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
            match[k - 1] += sum((hc & rc).values())
            tp[k - 1] += sum(hc.values())
            tr[k - 1] += sum(rc.values())
    P = statistics.fmean(match[k] / tp[k] if tp[k] else 0.0 for k in range(n))
    R = statistics.fmean(match[k] / tr[k] if tr[k] else 0.0 for k in range(n))
    return 100 * (1 + beta ** 2) * P * R / (beta ** 2 * P + R) if P + R else 0.0


def tico_pairs(root: Path, n: int) -> list[tuple[str, str]]:
    """English-Oromo sentence pairs from the TICO-19 dev/test TSVs (downloaded once if absent)."""
    import random
    import urllib.request
    import zipfile
    base = root / "data/raw/tico19_eval"
    pat = re.compile(r"en[-_](om|orm|gaz)\b", re.I)
    files = [f for f in root.glob("data/raw/**/*.tsv") if pat.search(f.name)]
    if not files:
        base.mkdir(parents=True, exist_ok=True)
        z = base / "tico19-testset.zip"
        if not z.exists():
            urllib.request.urlretrieve("https://tico-19.github.io/data/tico19-testset.zip", z)
        zipfile.ZipFile(z).extractall(base)
        files = [f for f in base.glob("**/*.tsv") if pat.search(f.name)]
    files = sorted(files, key=lambda f: ("dev" not in f.name, f.name))
    pairs = []
    with files[0].open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            en, om = (r.get("sourceString") or "").strip(), (r.get("targetString") or "").strip()
            if en and om:
                pairs.append((en, om))
    random.Random(3).shuffle(pairs)
    print(f"E3 data: {files[0].relative_to(root)} -> {min(n, len(pairs))} pairs", flush=True)
    return pairs[:n]


def cmd_mt(args: argparse.Namespace) -> None:
    from gadaabot.eval.ablation import NLLB, TR_SYS
    from gadaabot.agent.llm import JUDGE
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs = tico_pairs(args.root, args.mt_n)
    en, om = [p[0] for p in pairs], [p[1] for p in pairs]
    hyp: dict[str, tuple[list[str], list[str]]] = {}
    for sysname in args.mt_systems:
        if sysname == "nllb":
            nl = NLLB(args.nllb, args.nllb_device)
            hyp["nllb"] = (nl.translate(om, "gaz_Latn", "eng_Latn"), nl.translate(en, "eng_Latn", "gaz_Latn"))
        else:
            llm = LLM(args.llm_url, args.llm_model)

            def tr(text: str, to_en: bool, llm: LLM = llm) -> str:
                msg = (f"Translate this Afaan Oromoo text into English:\n\n{text}" if to_en
                       else f"Translate this English text into Afaan Oromoo (Qubee script):\n\n{text}")
                return llm.chat([{"role": "system", "content": TR_SYS}, {"role": "user", "content": msg}], max_tokens=400, sampling=JUDGE)
            with ThreadPoolExecutor(8) as ex:
                to_en = list(ex.map(lambda t: tr(t, True), om))
                to_om = list(ex.map(lambda t: tr(t, False), en))
            hyp[sysname] = (to_en, to_om)
    rows = []
    for sysname, (h_en, h_om) in hyp.items():
        def boot(h: list[str], r: list[str]) -> tuple[float, float]:
            import random
            rng = random.Random(5)
            vals = sorted(chrf([h[i] for i in idx], [r[i] for i in idx]) for idx in
                          ([rng.randrange(len(h)) for _ in range(len(h))] for _ in range(300)))
            return vals[7], vals[292]
        a, b = chrf(h_en, en), chrf(h_om, om)
        (alo, ahi), (blo, bhi) = boot(h_en, en), boot(h_om, om)
        rows.append({"system": sysname, "om_to_en_chrF": round(a, 1), "om_to_en_lo": round(alo, 1), "om_to_en_hi": round(ahi, 1),
                     "en_to_om_chrF": round(b, 1), "en_to_om_lo": round(blo, 1), "en_to_om_hi": round(bhi, 1), "n": len(pairs)})
        print("E3 " + " | ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
    write_tsv(out / f"E3_mt_{'_'.join(args.mt_systems)}.tsv", rows)


# ------------------------------------------------------------------------------------------- E7: retrieval components
ENCODERS = {  # name: (relative path under models root or shared, pooling, query template, passage template, trust_remote_code)
    "afrie5": ("McGill-NLP/AfriE5-Large-instruct", "mean", "Instruct: Given a question, retrieve the passage that answers the question\nQuery: {}", "{}", False),
    "me5_large_instruct": ("intfloat/multilingual-e5-large-instruct", "mean", "Instruct: Given a question, retrieve the passage that answers the question\nQuery: {}", "{}", False),
    "bge_m3": ("BAAI/bge-m3", "cls", "{}", "{}", False),
    "gte_multilingual": ("Alibaba-NLP/gte-multilingual-base", "cls", "{}", "{}", True),
    "labse": ("Sentence-Transformers/LaBSE", "pooler", "{}", "{}", False),
}


def bm25_index(texts: list[str], analyzer: Any, k1: float = 1.2, b: float = 0.75) -> Any:  # noqa: ANN401
    """Plain in-memory BM25 over child chunks; returns a scorer(query) -> {doc_index: score}."""
    import math
    from collections import Counter
    post: dict[str, list[tuple[int, int]]] = defaultdict(list)
    dl = []
    for i, t in enumerate(texts):
        terms = analyzer(t)
        dl.append(len(terms))
        for w, c in Counter(terms).items():
            post[w].append((i, c))
    n, avg = len(texts), statistics.fmean(dl)
    idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in post.items()}

    def score(q: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for w in set(analyzer(q)):
            for i, c in post.get(w, []):
                out[i] += idf[w] * c * (k1 + 1) / (c + k1 * (1 - b + b * dl[i] / avg))
        return out
    return score


def parent_order(child_ids: list[int], parents: list[str]) -> list[str]:
    seen, out = set(), []
    for c in child_ids:
        p = parents[c]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def rmetrics(ranked: dict[str, list[str]], gold: dict[str, str]) -> dict[str, Any]:
    r5 = [float(gold[q] in ranked[q][:5]) for q in gold]
    r1 = [float(gold[q] in ranked[q][:1]) for q in gold]
    mrr = [next((1 / (k + 1) for k, p in enumerate(ranked[q][:10]) if p == gold[q]), 0.0) for q in gold]
    m, lo, hi = mean_ci(r5)
    return {"R@1": round(100 * statistics.fmean(r1), 1), "R@5": round(100 * m, 1), "R@5_lo": round(100 * lo, 1),
            "R@5_hi": round(100 * hi, 1), "MRR@10": round(100 * statistics.fmean(mrr), 1), "_r5": dict(zip(gold, r5))}


def cmd_encoders(args: argparse.Namespace) -> None:
    import torch
    from qdrant_client import QdrantClient
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    from gadaabot.retrieval.oromo_analyzer import ANALYZERS, _trunc
    D = Data(args.root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    qs = [D.dev[i] for i in sorted(D.keep)]  # the 223 reviewed items, as in the final Gate R1
    gold = {d["id"]: d["parent_id"] for d in qs}
    qc = QdrantClient(url=args.qdrant_url, timeout=120)
    texts, parents, nxt = [], [], None
    while True:
        pts, nxt = qc.scroll(COLL, limit=2000, offset=nxt, with_payload=["text", "parent_id"], with_vectors=False)
        for p in pts:
            texts.append(str(p.payload["text"]))
            parents.append(str(p.payload["parent_id"]))
        if nxt is None:
            break
    print(f"E7 corpus: {len(texts)} child chunks, {len(qs)} reviewed questions (AfriE5 was chosen in T024, before review)", flush=True)
    dev = "cuda"
    rows: list[dict[str, Any]] = []
    # --- sparse: prefix-length sweep
    sparse_top: dict[str, dict[str, list[tuple[int, float]]]] = {}
    sparse_r5: dict[str, dict[str, float]] = {}
    for name, an in (("norm", ANALYZERS["norm"]), ("trunc3", _trunc(3)), ("trunc4", _trunc(4)), ("trunc5", _trunc(5)),
                     ("trunc6", _trunc(6)), ("stem", ANALYZERS["stem"])):
        sc = bm25_index(texts, an)
        tops = {d["id"]: sorted(sc(d["question"]).items(), key=lambda x: -x[1])[:100] for d in qs}
        sparse_top[name] = tops
        m = rmetrics({q: parent_order([c for c, _ in t], parents) for q, t in tops.items()}, gold)
        sparse_r5[name] = m["_r5"]
        rows.append({"component": f"BM25 {name}", **{k: v for k, v in m.items() if not k.startswith("_")}})
        print("E7 " + " | ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
    # --- reranker
    rtok = AutoTokenizer.from_pretrained(str(args.reranker))
    rr = AutoModelForSequenceClassification.from_pretrained(str(args.reranker), dtype=torch.float16).to(dev).eval()

    @torch.no_grad()
    def rerank(q: str, cands: list[int]) -> list[int]:
        b = rtok([q] * len(cands), [texts[c] for c in cands], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        s = rr(**b).logits.view(-1).float().cpu().tolist()
        return [c for _, c in sorted(zip(s, cands), key=lambda x: -x[0])]

    def rrf(lists: list[list[int]], k: int = 60) -> list[int]:
        f: dict[int, float] = defaultdict(float)
        for lst in lists:
            for r, c in enumerate(lst):
                f[c] += 1 / (k + r + 1)
        return [c for c, _ in sorted(f.items(), key=lambda x: -x[1])]
    full_r5: dict[str, dict[str, float]] = {}
    hyb_r5: dict[str, dict[str, float]] = {}
    dense_r5: dict[str, dict[str, float]] = {}
    for name, (rel, pool, qt, pt, trc) in ENCODERS.items():
        path = next((p for p in (args.root / "models" / rel, Path(args.shared_models) / rel) if (p / "config.json").exists()), None)
        if path is None:
            print(f"E7 skip {name}: not found ({rel})", flush=True)
            continue
        try:
            tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trc)
            enc = AutoModel.from_pretrained(str(path), dtype=torch.float16, trust_remote_code=trc).to(dev).eval()
        except Exception as e:  # noqa: BLE001  (one broken encoder must not kill the sweep)
            print(f"E7 FAIL {name} load: {type(e).__name__}: {str(e)[:300]}", flush=True)
            torch.cuda.empty_cache()
            continue

        @torch.no_grad()
        def embed(batch: list[str], enc: Any = enc, tok: Any = tok, pool: str = pool) -> Any:  # noqa: ANN401
            b = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            o = enc(**b)
            if pool == "cls":
                v = o.last_hidden_state[:, 0]
            elif pool == "pooler":
                v = o.pooler_output
            else:
                m = b["attention_mask"].unsqueeze(-1).to(o.last_hidden_state.dtype)
                v = (o.last_hidden_state * m).sum(1) / m.sum(1)
            return torch.nn.functional.normalize(v.float(), dim=-1).half()
        t0 = time.time()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        try:
            P = torch.empty((len(texts), enc.config.hidden_size), dtype=torch.float16, device=dev)
            for s0 in range(0, len(order), 256):
                idx = order[s0:s0 + 256]
                P[idx] = embed([pt.format(texts[i]) for i in idx])
            Q = torch.cat([embed([qt.format(d["question"]) for d in qs[s0:s0 + 64]]) for s0 in range(0, len(qs), 64)])
        except Exception as e:  # noqa: BLE001
            print(f"E7 FAIL {name} embed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            del enc
            torch.cuda.empty_cache()
            continue
        sims = (Q.float() @ P.float().T)
        dense_top = {d["id"]: sims[k].topk(100).indices.tolist() for k, d in enumerate(qs)}
        m_d = rmetrics({q: parent_order(t, parents) for q, t in dense_top.items()}, gold)
        hyb = {d["id"]: rrf([dense_top[d["id"]], [c for c, _ in sparse_top["trunc4"][d["id"]]]]) for d in qs}
        m_h = rmetrics({q: parent_order(t, parents) for q, t in hyb.items()}, gold)
        full = {d["id"]: rerank(d["question"], hyb[d["id"]][:50]) for d in qs}
        m_f = rmetrics({q: parent_order(t, parents) for q, t in full.items()}, gold)
        full_r5[name] = m_f["_r5"]
        hyb_r5[name] = m_h["_r5"]
        dense_r5[name] = m_d["_r5"]
        for lab, m in (("dense", m_d), ("dense+BM25trunc4 RRF", m_h), ("RRF + rerank50", m_f)):
            rows.append({"component": f"{name} {lab}", **{k: v for k, v in m.items() if not k.startswith("_")}})
            print("E7 " + " | ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
        print(f"E7 {name} embedding time {time.time() - t0:.0f}s", flush=True)
        del enc, P, sims
        torch.cuda.empty_cache()
    def pair_line(stage: str, a: str, b: str, ra: dict[str, float], rb: dict[str, float]) -> None:
        p = paired(ra, rb)
        print(f"E7-PAIR {stage} R@5 {a} - {b}: {100 * p['diff']:+.1f} pts [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}] p={p['p']}", flush=True)
    for name in ("norm", "trunc3", "trunc5", "trunc6", "stem"):
        pair_line("BM25", name, "trunc4", sparse_r5[name], sparse_r5["trunc4"])
    if "afrie5" in full_r5:
        for stage, store in (("dense", dense_r5), ("hybrid", hyb_r5), ("full", full_r5)):
            for name, r in store.items():
                if name != "afrie5":
                    pair_line(stage, name, "afrie5", r, store["afrie5"])
    print("E7 pre-registered: none (descriptive). Claim to check: prefix truncation is the best sparse analyzer and "
          "AfriE5 is at least as good as general multilingual encoders in the full pipeline.", flush=True)
    write_tsv(out / "E7_retrieval_components.tsv", rows)


# ------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["judge-inputs", "abstain", "tierfilter", "seeds", "rejudge", "mt", "encoders"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--llm-url", default="http://127.0.0.1:48000")
    ap.add_argument("--llm-model", default="gadaabot-gen")
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--tag", default="afriqueqwen9b")
    ap.add_argument("--test-control", action="store_true")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:46333")
    ap.add_argument("--afrie5", type=Path)
    ap.add_argument("--reranker", type=Path)
    ap.add_argument("--parents", type=Path)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed-list", type=int, nargs="+", default=[101, 202])
    ap.add_argument("--arms-dir", type=Path)
    ap.add_argument("--items-jsonl", type=Path, default=None, help="rejudge: question file instead of the 184 test items")
    ap.add_argument("--arm-list", nargs="+", default=BASE_ARMS)
    ap.add_argument("--mt-systems", nargs="+", default=["self"])
    ap.add_argument("--mt-n", type=int, default=300)
    ap.add_argument("--nllb", type=Path)
    ap.add_argument("--nllb-device", default="cuda:0")
    ap.add_argument("--shared-models", default="$SHARED_MODELS")
    args = ap.parse_args()
    args.judge_url = args.judge_url or args.llm_url
    args.judge_model = args.judge_model or args.llm_model
    t0 = time.time()
    {"judge-inputs": cmd_judge_inputs, "abstain": cmd_abstain, "tierfilter": cmd_tierfilter, "seeds": cmd_seeds,
     "rejudge": cmd_rejudge, "mt": cmd_mt, "encoders": cmd_encoders}[args.cmd](args)
    print(f"DONE {args.cmd} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
