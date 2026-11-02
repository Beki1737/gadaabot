"""GadaaBot 2.0 agent: a stateful, cyclic LangGraph that answers natively in Afaan Oromoo.

Topology
    ingress -> refine -> retrieve -> grade --(insufficient, attempts < MAX)--> reformulate -> retrieve
                                       |--(insufficient, attempts = MAX)--> abstain -> END
                                       |--(sufficient)--> generate -> verify --(too many unsupported,
                                                                           first time)--> generate
                                                                       |--> finalize -> END
    ingress --(emergency wording)--> emergency -> END

Nodes
  ingress      rule-based safety screen (danger-sign wording -> urgent-care message)
  refine       turns the user message into a standalone Oromo search query + domain
               (health / education / other); never translates to English
  retrieve     production retriever (hybrid DBSF + rerank + parents); authority prior for health
  grade        LLM judge: do the passages answer the question? which are useful? if not,
               what is missing and a better Oromo query
  reformulate  adopt the judge's query (bounded loop: MAX_ATTEMPTS searches)
  generate     answer in Afaan Oromoo from the useful passages only, citing [n]
  verify       LLM judge per sentence: supported by its cited passages?
               unsupported sentences are removed; > 40% unsupported -> one stricter regeneration
  finalize     source list + domain disclaimer (health) appended
  abstain      honest "not found in reliable sources" message with safe general advice
All Oromo user-facing fixed texts are drafts for native-speaker review (OROMO_TEXTS).

v5 (T032): the echo rule only fires on a real prompt echo ('Passages:' followed by a '[n] (source;'
header); in T030 it also discarded answers that merely began with the word 'Passages [1] akka jedhuutti'
(18/316 naive-RAG items counted as abstentions). A real echo now triggers one regeneration in every
mode, and the English words 'passage(s)' are rewritten as 'madda/maddeewwan'.
v4: build_graph(use_refine, use_grade, use_verify) switches nodes off for the ablation
(naive RAG = retrieve -> generate -> finalize). Bug fix: a model-appended source list no longer
marks the whole draft as an echo (T029: 'generate (0 words)' on chat1, dev0096); the list is
stripped instead. Raw drafts are kept in the state for diagnostics.
"""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from gadaabot.agent.llm import LLM
from gadaabot.agent.retriever import Evidence, Retriever

MAX_ATTEMPTS = 2
RERANK_TRUST = 2.5      # cross-encoder score above which a 'no' from the LLM grader is overridden to 'partly'
FALLBACK_MIN = 2.0      # minimum cross-encoder score for the extractive fallback; below it the agent abstains
_DEBRIS = re.compile(r"_{3,}|\.{5,}|(?:\b\d+\s*){6,}")
OROMO_TEXTS = {
    "disclaimer_health": "Hubachiisa: Odeeffannoon kun barumsaaf qofa kennameera; gorsa ogeessa fayyaa bakka hin bu'u. "
                         "Mallattoon cimaan yoo jiraate, hatattamaan buufata fayyaa dhiyoo jirutti deemaa.",
    "sources": "Maddeewwan",
    "abstain": "Dhiifama, gaaffii kanaaf odeeffannoo amanamaa ta'e maddeewwan koo keessatti hin arganne. "
               "Odeeffannoo sirrii argachuuf ogeessa fayyaa ykn barsiisaa keessan gaafachuu ni dandeessu.",
    "emergency": "Kun haala ariifachiisaa ta'uu danda'a. Hatattamaan buufata fayyaa ykn hospitaala dhiyoo jirutti "
                 "deemaa ykn tajaajila ambulaansii bilbilaa. Namni dhukkubsate kophaa isaa hin dhiisinaa.",
}
# danger-sign wording (draft list for native and clinical review)
EMERGENCY = re.compile(
    r"hafuura (baafachuu|fudhachuu) (hin danda'|dadhab)|hafuurri (na|isa|ishee|ishii) (cite|dhabam)|"
    r"of ajjees|of ajjeesuu|lubbuu (koo|isaa|ishee) baasuu|dhiigni (baay'een|hedduun|dhaabbachuu hin)|"
    r"of (wallaale|wallaaltee|wallaalaa)|(sammuu|yaada) (dhabe|dhabde)|qabbanaa'e hin socho'u|"
    r"(garaa|mataa) dhukkubbii (cimaa|hamaa) .*ulfa|ulfa .*dhiiga|daa'imni .*(hin hargansu|hafuura hin)", re.I)

SYS_REFINE = ("You prepare search queries for an Afaan Oromoo health and education knowledge base. "
              "Work in Afaan Oromoo only; never translate into English.")
SYS_JUDGE = "You are a strict evaluator. Read Afaan Oromoo carefully. Reply with JSON only."
SYS_ANSWER = (
    "You are GadaaBot, a helpful assistant for Afaan Oromoo speakers in health and education. "
    "Write ONLY in Afaan Oromoo (Qubee). Use ONLY facts stated in the numbered passages and put the passage "
    "number in brackets, like [1] or [2], after every sentence that uses it. If the passages do not cover part "
    "of the question, say so plainly instead of guessing. Be clear, practical and kind; explain simply when the "
    "user is a student. Never invent facts, never give medicine doses, and never contradict the passages. "
    "Explain in your own words instead of copying long passages; match the level the user asks for (for "
    "example a grade-6 student needs simple words and short sentences). Do not repeat the question and never "
    "print the passages themselves. Cite only as [n]. Never use the English word 'passage'.")


class State(TypedDict, total=False):
    question: str
    emergency: bool
    domain: str
    query: str
    attempts: int
    evidence: list[Evidence]
    context: list[Evidence]
    useful: list[int]
    sufficient: bool
    missing: str
    draft: str
    regenerated: bool
    second_pass: bool
    answer: str
    sources: list[dict[str, str]]
    abstained: bool
    abstain_reason: str
    trace: list[str]
    drafts: list[str]


def _passages(ev: list[Evidence]) -> str:
    return "\n\n".join(f"[{i + 1}] ({e.source}; {e.title[:80]})\n{e.text}" for i, e in enumerate(ev))


_ECHO = re.compile(r"^\s*(passages?|question|gaaffii)\s*:|^\s*\[\d+\]\s*\(", re.I)
_PROMPT_ECHO = re.compile(r"\s*passages?\s*:?\s*\n+\s*\[\d+\]\s*\(", re.I)
_ENGLISH_PASSAGE = re.compile(r"\bpassages?\b", re.I)
_SRC_TAIL = re.compile(r"\n\s*(?:[#*_]+\s*)?(?:Maddeewwan|Madda|Wabii|Sources?|References?|Passages?)\b[^\n]*\n(?:\s*(?:[-*]\s*)?\[?\d+[\].)][^\n]*(?:\n|$))+\s*$", re.I)
_CITE_VARIANTS = re.compile(r"\(?\s*(?:passage|madda|kutaa)\s*\[?(\d+)\]?\s*\)?", re.I)


def sanitize(draft: str) -> str:
    """Drop echoed prompt lines and normalize citation variants like '(Passage [2])' to '[2]'."""
    draft = _SRC_TAIL.sub("", draft.rstrip() + "\n")  # source list appended by the model
    lines = [l for l in draft.splitlines() if not _ECHO.search(l)]
    text = "\n".join(lines).strip()
    return _CITE_VARIANTS.sub(lambda m: f"[{m.group(1)}]", text)


def _sentences(text: str) -> list[str]:
    """Split into sentences; a citation group after the full stop ('... dha. [1][2]') stays with its sentence,
    and fragments without at least 3 words (e.g. a lone '[1]' or a list number) are dropped."""
    text = re.sub(r"([.!?])\s+((?:\[\d+\]\s*)+)", r" \2\1 ", text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    return [p for p in parts if len(re.findall(r"[^\W\d_]{2,}", p)) >= 3]


def _content_words(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[^\W\d_]{3,}", text.lower())}


def build_graph(llm: LLM, retriever: Retriever, authority_boost: float = 1.0, use_refine: bool = True,  # type: ignore[no-untyped-def]
                use_grade: bool = True, use_verify: bool = True):
    from langgraph.graph import END, StateGraph

    def log(s: State, msg: str) -> list[str]:
        return (s.get("trace") or []) + [msg]

    def ingress(s: State) -> State:
        emer = bool(EMERGENCY.search(s["question"]))
        return {"emergency": emer, "attempts": 0, "regenerated": False, "abstained": False, "query": s["question"],
                "domain": "", "drafts": [], "trace": log(s, f"ingress emergency={emer}")}

    def emergency(s: State) -> State:
        return {"answer": OROMO_TEXTS["emergency"], "sources": [], "trace": log(s, "emergency message")}

    def refine(s: State) -> State:
        out = llm.chat_json([{"role": "system", "content": SYS_REFINE}, {"role": "user", "content": (
            f"User message: {s['question']}\n\nReturn JSON: {{\"query\": \"<standalone Afaan Oromoo search query "
            f"containing the key terms>\", \"domain\": \"health|education|other\"}}")}])
        q = str(out.get("query") or s["question"]).strip()
        dom = str(out.get("domain") or "other").strip().lower()
        dom = dom if dom in ("health", "education", "other") else "other"
        return {"query": q, "domain": dom, "trace": log(s, f"refine domain={dom} query={q!r}")}

    def retrieve(s: State) -> State:
        boost = authority_boost if s.get("domain") == "health" else 0.0
        queries = [s["query"]] if s.get("attempts", 0) else list(dict.fromkeys([s["question"], s["query"]]))
        pool: dict[str, Evidence] = {}
        for q in queries:  # original wording + refined query (first search); refined only on retries
            for e in retriever.search(q, k_parents=4, authority_boost=boost):
                if e.parent_id not in pool or e.score > pool[e.parent_id].score:
                    pool[e.parent_id] = e
        ev = sorted(pool.values(), key=lambda e: -e.score)[:4]
        return {"evidence": ev, "attempts": s.get("attempts", 0) + 1,
                "trace": log(s, f"retrieve #{s.get('attempts', 0) + 1}: " +
                             ", ".join(f"{e.source}:{e.score:+.1f}" for e in ev))}

    def grade(s: State) -> State:
        ev = s.get("evidence") or []
        if not ev:
            return {"sufficient": False, "useful": [], "missing": "", "trace": log(s, "grade: no evidence")}
        out = llm.chat_json([{"role": "system", "content": SYS_JUDGE}, {"role": "user", "content": (
            f"Question: {s['question']}\n\nPassages:\n{_passages(ev)}\n\n"
            "Can the question be answered from these passages? 'yes' = fully; 'partly' = the main point is "
            "covered but some details are missing; 'no' = the passages are about something else. Return JSON: "
            "{\"answerable\": \"yes\" or \"partly\" or \"no\", \"useful\": [numbers of the passages that help], "
            "\"missing\": \"<what is missing, in Afaan Oromoo>\", "
            "\"better_query\": \"<a better Afaan Oromoo search query if not answerable>\"}")}], max_tokens=300)
        useful = [int(i) for i in out.get("useful", []) if str(i).isdigit() and 1 <= int(i) <= len(ev)]
        level = str(out.get("answerable", "no")).strip().lower()
        top = max((e.score for e in ev), default=-99.0)
        if level == "no" and top >= RERANK_TRUST:  # the cross-encoder is confident: answer partially, state the gap
            level, useful = "partly", useful or [1]
        suff = level in ("yes", "partly") and bool(useful)
        upd: State = {"sufficient": suff, "useful": useful or list(range(1, len(ev) + 1)),
                      "missing": str(out.get("missing", "")) if level == "partly" else "",
                      "trace": log(s, f"grade answerable={level} useful={useful}")}
        if not suff and out.get("better_query"):
            upd["query"] = str(out["better_query"]).strip()
        return upd

    def reformulate(s: State) -> State:
        return {"trace": log(s, f"reformulate -> {s.get('query')!r}")}

    def generate(s: State) -> State:
        allev = s.get("evidence") or []
        ev = [allev[i - 1] for i in s.get("useful", []) if 1 <= i <= len(allev)] or allev
        strict = " Use fewer, shorter sentences and only what the passages say explicitly." if s.get("regenerated") else ""
        gap = (f"\n\nNote: the passages do not cover: {s['missing']}. Say briefly that this part is not covered."
               if s.get("missing") else "")
        msgs = [{"role": "system", "content": SYS_ANSWER + strict}, {"role": "user", "content": (
            f"Passages:\n{_passages(ev)}\n\nQuestion: {s['question']}{gap}")}]
        draft = llm.chat(msgs, max_tokens=700)
        raw = draft
        if _PROMPT_ECHO.match(draft):  # the model copied the passage block: one retry with an explicit reminder
            draft = llm.chat(msgs + [{"role": "assistant", "content": draft[:300]}, {"role": "user", "content":
                             "Do not copy the passages. Answer the question in your own words in Afaan Oromoo, citing [n]."}],
                             max_tokens=700)
            raw = raw[:700] + "\n<<RETRY>>\n" + draft
        echo = bool(_PROMPT_ECHO.match(draft))
        draft = "" if echo else _ENGLISH_PASSAGE.sub(lambda m: "maddeewwan" if m.group(0).lower().endswith("s") else "madda",
                                                     sanitize(draft))
        draft = draft[:1].upper() + draft[1:]
        upd: State = {"draft": draft, "context": ev, "second_pass": bool(s.get("regenerated")),
                      "drafts": (s.get("drafts") or []) + [raw[:1500]],
                      "trace": log(s, f"generate ({len(draft.split())} words{', strict' if strict else ''})")}
        if not use_verify:
            upd["answer"] = draft
        return upd

    def verify(s: State) -> State:
        sents = _sentences(s.get("draft", ""))
        if not sents:
            if not s.get("regenerated"):
                return {"regenerated": True, "answer": "", "trace": log(s, "verify: empty or echoed draft -> regenerate")}
            return {"answer": extractive(s), "trace": log(s, "verify: empty draft again -> extractive fallback")}
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sents))
        out = llm.chat_json([{"role": "system", "content": SYS_JUDGE}, {"role": "user", "content": (
            f"Passages:\n{_passages(s['context'])}\n\nAnswer sentences:\n{numbered}\n\n"
            "Mark a sentence as unsupported ONLY if it states a fact that is absent from the passages or contradicts "
            "them. Paraphrases, simplifications, summaries, greetings and statements that information is missing "
            "count as supported. Return JSON: {\"unsupported\": [numbers of unsupported sentences]}")}], max_tokens=200)
        bad = {int(i) for i in out.get("unsupported", []) if str(i).isdigit() and 1 <= int(i) <= len(sents)}
        frac = len(bad) / len(sents)
        if frac > 0.4 and not s.get("regenerated"):
            return {"regenerated": True, "answer": "", "trace": log(s, f"verify: {len(bad)}/{len(sents)} unsupported -> regenerate")}
        qw = _content_words(s["question"])
        kept = [t for i, t in enumerate(sents) if i + 1 not in bad and not (
            t.rstrip().endswith("?") and len(_content_words(t) - qw) <= 1)]  # question echo
        if sum(len(t.split()) for t in kept) < 6:
            kept = []
        if not kept:
            return {"answer": extractive(s), "trace": log(s, f"verify: all {len(sents)} unsupported -> extractive fallback")}
        return {"answer": " ".join(kept), "trace": log(s, f"verify: removed {len(bad)}/{len(sents)} unsupported sentences")}

    def extractive(s: State) -> str:
        """Grounded fallback: quote the best-matching chunks of the top passages with citations."""
        ctx = s.get("context") or []
        parts = [f"{e.child.strip()} [{i + 1}]" for i, e in enumerate(ctx[:2])
                 if e.child.strip() and e.score >= FALLBACK_MIN and not _DEBRIS.search(e.child)]
        return ("Akka maddeewwan armaan gadii jedhanitti: " + " ".join(parts)) if parts else ""

    def finalize(s: State) -> State:
        if not s.get("answer"):
            return abstain({**s, "abstain_reason": "no_supported_answer"})
        ev = s.get("context") or []
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", s["answer"]) if 1 <= int(n) <= len(ev)})
        src = [{"n": str(n), "title": ev[n - 1].title, "source": ev[n - 1].source, "url": ev[n - 1].url,
                "tier": ev[n - 1].tier} for n in (cited or range(1, len(ev) + 1))]
        text = s["answer"] + "\n\n" + OROMO_TEXTS["sources"] + ":\n" + "\n".join(
            f"[{x['n']}] {x['title']} ({x['source']})" for x in src)
        if s.get("domain") == "health":
            text += "\n\n" + OROMO_TEXTS["disclaimer_health"]
        return {"answer": text, "sources": src, "trace": log(s, f"finalize cited={cited}")}

    def abstain(s: State) -> State:
        reason = s.get("abstain_reason") or ("no_evidence" if not s.get("evidence") else "evidence_judged_insufficient")
        return {"answer": OROMO_TEXTS["abstain"], "sources": [], "abstained": True, "abstain_reason": reason,
                "trace": log(s, f"abstain ({reason})")}

    g = StateGraph(State)
    for name, fn in (("ingress", ingress), ("emergency", emergency), ("refine", refine), ("retrieve", retrieve),
                     ("grade", grade), ("reformulate", reformulate), ("generate", generate), ("verify", verify),
                     ("finalize", finalize), ("abstain", abstain)):
        g.add_node(name, fn)
    g.set_entry_point("ingress")
    first = "refine" if use_refine else "retrieve"
    g.add_conditional_edges("ingress", lambda s: "emergency" if s.get("emergency") else first,
                            {"emergency": "emergency", first: first})
    g.add_edge("emergency", END)
    g.add_edge("refine", "retrieve")
    if use_grade:
        g.add_edge("retrieve", "grade")
    else:
        g.add_edge("retrieve", "generate")
    g.add_conditional_edges("grade", lambda s: "generate" if s.get("sufficient") else (
        "reformulate" if s.get("attempts", 0) < MAX_ATTEMPTS else "abstain"),
        {"generate": "generate", "reformulate": "reformulate", "abstain": "abstain"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", "verify" if use_verify else "finalize")
    g.add_conditional_edges("verify", lambda s: "generate" if (s.get("regenerated") and not s.get("second_pass")
                                                                 and not s.get("answer")) else "finalize",
                            {"generate": "generate", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)
    return g.compile()


def answer(app: Any, question: str) -> dict[str, Any]:  # noqa: ANN401
    t0 = time.time()
    out = app.invoke({"question": question}, {"recursion_limit": 30})
    return {"question": question, "answer": out.get("answer", ""), "sources": out.get("sources", []),
            "domain": out.get("domain", ""), "abstained": bool(out.get("abstained")),
            "abstain_reason": out.get("abstain_reason", ""), "emergency": bool(out.get("emergency")),
            "attempts": out.get("attempts", 0), "trace": out.get("trace", []), "seconds": round(time.time() - t0, 2),
            "evidence_ids": [e.parent_id for e in (out.get("evidence") or [])],
            "evidence_scores": [e.score for e in (out.get("evidence") or [])],
            "contexts": [e.text for e in (out.get("context") or out.get("evidence") or [])],
            "drafts": out.get("drafts", []),
            "evidence_domains": [e.domain for e in (out.get("context") or out.get("evidence") or [])]}
