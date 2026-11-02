"""GadaaBot knowledge base, step 1: acquire Afaan Oromo health / education documents with
full provenance and a license tier per document.

License tiers (decision D39):
  A  open and redistributable: CC0, CC BY, CC BY-SA, public domain
  B  open, non-commercial: CC BY-NC(-SA)
  C  publicly posted without an open license: indexed locally for research evaluation
     only; text is never redistributed (release = URLs + this script)

Sources in this step (each runs independently; one failure does not stop the others):
  wikipedia   Oromo Wikipedia (wikimedia/wikipedia 20231101.om), CC BY-SA 4.0       A
  tico19      TICO-19 COVID-19 translations, Oromo side, CC0                          A
  storybook   GlotStoryBook (African Storybook etc.), Oromo 'orm', per-file CC        A/B
  ohkb        OHKB Vol. 1 health-evidence summaries in Afaan Oromo, CC BY 4.0         A
  mdh         Minnesota Department of Health translated materials (Oromo PDFs)        C

Output: data/kb/raw/<source>.jsonl.gz (one JSON document per line) and
data/kb/raw/manifest.json (per-source counts, words, Oromo-check pass rate, license).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from gadaabot.data.qubee import non_qubee_word_ratio, normalize_qubee, profile, words

UA = "GadaaBot-research/0.1 (academic research; low-rate, robots-respecting)"
MDH_PAGES = ("https://www.health.state.mn.us/communities/translation/oromo.html",
             "https://www.health.state.mn.us/communities/rih/hlthmat.html")
OHKB_URL = "https://ohkb.org/papers/ohkb-vol1-translating-health-evidence-oromia.pdf"


@dataclass
class KBDoc:
    doc_id: str
    source: str
    tier: str
    license: str
    url: str
    title: str
    domain: str
    text: str
    n_words: int = 0
    oromo_ok: bool = False
    retrieved: str = ""
    para_filtered: bool = False   # True: non-Oromo paragraphs were removed (bilingual PDFs)


_SENT = re.compile(r"(?<=[.?!])\s+")
_OROMO_CUE = re.compile(r"aa|ii|uu|dh|'")


def is_oromo_sent(text: str) -> bool:
    """Sentence-level language check for mixed PDFs. Qubee script required; English or
    Somali function words must not dominate; a sentence with no function words at all is
    kept only if it carries Qubee orthographic cues (long vowels aa/ii/uu, dh, hudhaa)."""
    ws = words(text)
    if len(ws) < 2 or non_qubee_word_ratio(ws) > 0.05:
        return False
    p = profile(text)
    if p.english_fw > p.oromo_fw or p.somali_fw > p.oromo_fw + 0.05 or p.english_fw >= 0.15:
        return False
    if p.oromo_fw == 0 and p.english_fw == 0:
        return sum(bool(_OROMO_CUE.search(w)) for w in ws) / len(ws) >= 0.25
    return True


def oromo_paragraphs(text: str) -> str:
    """Keep only Oromo sentences of a (possibly bilingual) PDF text. Blocks end at blank
    lines; within a block, sentences are classified one by one and consecutive Oromo
    sentences are re-joined, so a bilingual page keeps exactly its Oromo part."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        flat = " ".join(l.strip() for l in block.splitlines() if l.strip())
        run: list[str] = []
        for sent in _SENT.split(flat):
            if is_oromo_sent(sent):
                run.append(sent)
            elif run:
                out.append(" ".join(run))
                run = []
        if run:
            out.append(" ".join(run))
    return "\n\n".join(out)


def finalize(d: KBDoc) -> KBDoc:
    d.text = normalize_qubee(d.text)
    if d.para_filtered:
        d.text = oromo_paragraphs(d.text)
    ws = words(d.text)
    p = profile(d.text)
    d.n_words = len(ws)
    # Oromo check: function words dominate foreign ones and the text is Qubee (ASCII) script
    d.oromo_ok = bool(ws) and non_qubee_word_ratio(ws) <= 0.05 and (
        p.looks_oromo or (d.para_filtered and p.english_fw < 0.05))
    d.doc_id = d.doc_id or hashlib.sha1(f"{d.source}|{d.url}|{d.text[:200]}".encode()).hexdigest()[:16]
    d.retrieved = time.strftime("%Y-%m-%d")
    return d


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def pdf_text(data: bytes) -> tuple[str, int]:
    import fitz  # pymupdf

    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = [p.get_text("text") for p in doc]
    return "\n\n".join(pages), len(pages)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def src_wikipedia(_: Path) -> Iterator[KBDoc]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.om", split="train")
    for r in ds:
        yield KBDoc(f"wiki-{r['id']}", "wikipedia", "A", "CC BY-SA 4.0", r["url"], r["title"], "encyclopedia",
                    r["text"])


def src_tico19(work: Path) -> Iterator[KBDoc]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files("gmnlp/tico19", repo_type="dataset")
             if re.search(r"(^|[/_.-])(en-om|om-en)([/_.-]|$)", f)]
    if not files:
        raise RuntimeError("no en-om files found in gmnlp/tico19")
    by_url: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in files:
        p = hf_hub_download("gmnlp/tico19", f, repo_type="dataset", cache_dir=str(work / "hf"))
        for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) < 6 or parts[0] == "sourceLang":
                continue
            src_lang, tgt_lang, _src, tgt, sid, url = parts[:6]
            if tgt_lang.strip().lower().startswith("om"):
                by_url[url.strip() or "tico19"].append((sid, tgt))
    for url, sents in by_url.items():
        sents.sort()
        yield KBDoc("", "tico19", "A", "CC0 1.0", url, f"TICO-19: {url[:80]}", "health",
                    " ".join(s for _, s in sents))


def src_storybook(_: Path) -> Iterator[KBDoc]:
    from datasets import load_dataset

    ds = load_dataset("cis-lmu/GlotStoryBook", "default", split="train")
    cols = ds.column_names
    lang_col = next((c for c in cols if c.lower() in ("iso639-3", "iso639_3", "iso", "language", "lang", "iso_code")),
                    None)
    if lang_col is None:  # find the column that contains 'orm' values
        lang_col = next(c for c in cols if "orm" in set(map(str, ds.select(range(min(len(ds), 5000)))[c])))
    text_col = next(c for c in cols if c.lower() in ("text", "story", "content"))
    lic_col = next((c for c in cols if "licen" in c.lower()), None)
    title_col = next((c for c in cols if "title" in c.lower()), None)
    src_col = next((c for c in cols if c.lower() in ("source", "publisher", "repo", "source_repo")), None)
    for i, r in enumerate(ds.filter(lambda x: str(x[lang_col]).startswith("orm"))):
        lic = str(r[lic_col]) if lic_col else "unknown"
        tier = "B" if "NC" in lic.upper() else ("A" if lic != "unknown" else "C")
        yield KBDoc(f"story-{i}", "storybook", tier, lic, str(r[src_col]) if src_col else "GlotStoryBook",
                    str(r[title_col]) if title_col else f"story {i}", "education", str(r[text_col]))


def src_ohkb(work: Path) -> Iterator[KBDoc]:
    data = http_get(OHKB_URL)
    (work / "ohkb_vol1.pdf").write_bytes(data)
    text, n_pages = pdf_text(data)
    lic = "CC BY 4.0" if re.search(r"CC[ -]BY[ -]4\.0|Creative Commons Attribution 4\.0", text, re.I) else "unverified"
    tier = "A" if lic == "CC BY 4.0" else "C"
    # keep the document whole; chunking happens at ingestion
    yield KBDoc("ohkb-vol1", "ohkb", tier, lic, OHKB_URL, f"OHKB Vol. 1 ({n_pages} pages)", "health", text,
                para_filtered=True)


def src_mdh(work: Path) -> Iterator[KBDoc]:
    pdf_dir = work / "mdh_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for page in MDH_PAGES:
        html = http_get(page).decode("utf-8", errors="replace")
        for href, label in re.findall(r'<a[^>]+href="([^"]+?\.pdf)"[^>]*>(.*?)</a>', html, re.I | re.S):
            url = urllib.parse.urljoin(page, href)
            lab = re.sub(r"<[^>]+>|\s+", " ", label).strip()
            if page.endswith("hlthmat.html") and "oro" not in url.lower() and "oromo" not in lab.lower():
                continue
            links.setdefault(url, lab)
    for url, label in sorted(links.items()):
        name = re.sub(r"[^A-Za-z0-9._-]", "_", urllib.parse.urlparse(url).path.split("/")[-1])
        f = pdf_dir / name
        if not f.exists():
            time.sleep(1.0)  # polite rate
            try:
                f.write_bytes(http_get(url))
            except Exception as exc:  # noqa: BLE001
                print(f"  mdh: skip {url} ({type(exc).__name__})", flush=True)
                continue
        try:
            text, _ = pdf_text(f.read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f"  mdh: unreadable {name} ({type(exc).__name__})", flush=True)
            continue
        yield KBDoc("", "mdh", "C", "no open license (State of Minnesota); research index only, not redistributed",
                    url, label or name, "health", text, para_filtered=True)


# ---------------------------------------------------------------------------
# polite PDF crawler (robots.txt respected, same host, 1 request/s)
# ---------------------------------------------------------------------------
def crawl_pdfs(seeds: list[str], keep: "re.Pattern[str]", max_depth: int = 2, max_pages: int = 300,
               max_pdfs: int = 400) -> dict[str, str]:
    """Return {pdf_url: anchor text} reachable from the seeds whose URL or anchor matches keep."""
    import urllib.robotparser

    robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(url: str) -> bool:
        host = urllib.parse.urlparse(url)
        key = f"{host.scheme}://{host.netloc}"
        if key not in robots:
            rp = urllib.robotparser.RobotFileParser(key + "/robots.txt")
            try:
                rp.read()
                robots[key] = rp
            except Exception:  # noqa: BLE001  (no robots.txt reachable: allow)
                robots[key] = None
        rp = robots[key]
        return True if rp is None else rp.can_fetch(UA, url)

    pdfs: dict[str, str] = {}
    seen: set[str] = set()
    frontier = [(u, 0) for u in seeds]
    pages = 0
    while frontier and pages < max_pages and len(pdfs) < max_pdfs:
        url, depth = frontier.pop(0)
        if url in seen or not allowed(url):
            continue
        seen.add(url)
        if url.lower().split("?")[0].endswith(".pdf"):
            pdfs.setdefault(url, "")
            continue
        time.sleep(1.0)
        try:
            html = http_get(url, timeout=40).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  crawl: skip {url} ({type(exc).__name__})", flush=True)
            continue
        pages += 1
        host = urllib.parse.urlparse(url).netloc
        for href, label in re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\'][^>]*>(.*?)</a>', html, re.I | re.S):
            nxt = urllib.parse.urljoin(url, href.strip())
            if not nxt.startswith("http") or urllib.parse.urlparse(nxt).netloc != host:
                continue
            lab = re.sub(r"<[^>]+>|\s+", " ", label).strip()
            if nxt.lower().split("?")[0].endswith(".pdf"):
                if keep.search(nxt) or keep.search(lab):
                    pdfs.setdefault(nxt, lab)
            elif depth + 1 <= max_depth and nxt not in seen:
                frontier.append((nxt, depth + 1))
    print(f"  crawl: {pages} pages visited, {len(pdfs)} matching PDFs", flush=True)
    return pdfs


def pdf_docs(pdfs: dict[str, str], work: Path, sub: str, source: str, tier: str, lic: str, domain_of: Callable[[str], str],
             max_mb: int = 120) -> Iterator[KBDoc]:
    d = work / sub
    d.mkdir(parents=True, exist_ok=True)
    for url, label in sorted(pdfs.items()):
        name = hashlib.sha1(url.encode()).hexdigest()[:10] + "_" + re.sub(r"[^A-Za-z0-9._-]", "_", url.split("/")[-1])[-80:]
        f = d / name
        if not f.exists():
            time.sleep(1.0)
            try:
                data = http_get(url, timeout=180)
            except Exception as exc:  # noqa: BLE001
                print(f"  {source}: skip {url} ({type(exc).__name__})", flush=True)
                continue
            if len(data) > max_mb * 1024 * 1024 or not data.startswith(b"%PDF"):
                print(f"  {source}: not a PDF or too large: {url}", flush=True)
                continue
            f.write_bytes(data)
        try:
            text, n_pages = pdf_text(f.read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f"  {source}: unreadable {url} ({type(exc).__name__})", flush=True)
            continue
        title = label or urllib.parse.unquote(url.split("/")[-1])
        yield KBDoc("", source, tier, lic, url, f"{title} ({n_pages} pages)", domain_of(url + " " + title), text,
                    para_filtered=True)


TEXTBOOK_SEEDS = [
    "https://camaraethiopia.org.et/OromiaPrimary/CEE_MoE_Plasma/content/Environmental-Science/oro_index.PHP",
    "https://camaraethiopia.org.et/OromiaPrimary/",
    "http://213.55.101.25/index.php/leecalloo/book-library/266/showCategory/56/",
    "https://content.temari.et/textbooks/22015a42/Grade-1-Environmental-Science_Student-Textbook_Oromia_Afaan-Oromoo.pdf",
]
TEXTBOOK_KEEP = re.compile(r"oro|afaan|saayinsii|fayyaa|naannoo|waliigalaa|environmental|science|health|herrega|"
                           r"barnoota|kitaab|kutaa", re.I)


def src_textbooks(work: Path) -> Iterator[KBDoc]:
    pdfs = crawl_pdfs(TEXTBOOK_SEEDS, TEXTBOOK_KEEP, max_depth=2, max_pages=250)

    def dom(s: str) -> str:
        return "health" if re.search(r"fayyaa|health", s, re.I) else "education"

    yield from pdf_docs(pdfs, work, "textbooks_pdf", "textbooks", "C",
                        "(c) Biiroo Barnootaa Oromiyaa / Ethiopian MoE; research index only; permission requested", dom)


def src_ethnomed(work: Path) -> Iterator[KBDoc]:
    pdfs = crawl_pdfs(["https://ethnomed.org/resource/oromo-in-language-health-hub/",
                       "https://ethnomed.org/language/oromo/"], re.compile(r"oromo|afaan|oro", re.I),
                      max_depth=1, max_pages=120)
    yield from pdf_docs(pdfs, work, "ethnomed_pdf", "ethnomed", "C",
                        "EthnoMed (University of Washington) patient education; research index only", lambda s: "health")


# Oromo topic lexicons (heuristic; extend with native-speaker review). Stems match word prefixes.
HEALTH_TERMS = ("fayyaa", "dhukkub", "dhibee", "qoricha", "yaal", "hospitaal", "kilinik", "ogeessa fayyaa", "talaallii",
                "busaa", "garaa kaasaa", "ulfa", "da'umsa", "deessuu", "daa'ima", "nyaata madaalawaa", "soorata",
                "hiv", "eedsii", "vaayirasii", "koronaa", "covid", "kaansarii", "sukkaara", "dhiibbaa dhiigaa",
                "qulqullina", "bishaan qulqulluu", "mallattoo", "dawaa", "qaama", "tiibii", "sombaa", "onnee")
EDU_TERMS = ("barnoota", "barataa", "barattoota", "barsiisaa", "mana barumsaa", "yuunivarsiitii", "kolleejjii",
             "saayinsii", "herrega", "qorannoo", "qormaata", "kutaa", "sirna barnootaa", "kitaaba", "leenjii",
             "teeknooloojii", "fiiziksii", "keemistirii", "baayoloojii")


def topic_score(text: str, terms: tuple[str, ...]) -> tuple[int, float]:
    low = " " + " ".join(words(text)) + " "
    hits = [t for t in terms if (" " + t) in low]
    total = sum(low.count(" " + t) for t in hits)
    return len(hits), 100 * total / max(1, len(low.split()))


def src_webdomain(work: Path) -> Iterator[KBDoc]:
    """Health / education documents from corpus v3 (train + heldout), selected by topic lexicon."""
    interim = work.parents[2] / "interim"
    for split in ("train", "heldout"):
        path = interim / f"oromo_corpus.v3.{split}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["n_words"] < 80:
                    continue
                hd, hdens = topic_score(r["text"], HEALTH_TERMS)
                ed, edens = topic_score(r["text"], EDU_TERMS)
                if max(hd, ed) < 4 or max(hdens, edens) < 1.5:
                    continue
                dom = "health" if (hd, hdens) >= (ed, edens) else "education"
                yield KBDoc(f"web-{r['id']}", "webdomain", "C",
                            f"web text via research corpus ({r['source']}); research index only",
                            r.get("origin", r["source"]), f"{r['source']} {split} {r['id']}", dom, r["text"])


OEB_BASE = "http://213.55.101.25/images/new_book/"
OEB_SUBJECTS = ("Saayinsii_Naannoo", "Saayinsii_Waliigalaa", "FJQ", "HERREGA", "Herrega", "Og-Aartiiwwan",
                "Afan_Oromo", "Afaan_Oromoo", "Barnoota_Hawaasaa", "Saayinsii", "Fiiziksii", "Keemistirii",
                "Baayoloojii", "Teeknooloojii")


def src_oeb(work: Path) -> Iterator[KBDoc]:
    """Oromia Education Bureau school books (grades 1-8). T019 found grades 1, 3, 5 under
    /images/new_book/{grade1,g3,g5}/; this source first tries the directory listings, then
    probes the same file-name pattern for the missing grades (HEAD requests, 1 per second)."""
    found: dict[str, str] = {}
    dirs = [OEB_BASE] + [OEB_BASE + d + "/" for n in range(1, 9) for d in (f"g{n}", f"grade{n}", f"G{n}", f"Grade{n}")]
    for d in dirs:
        time.sleep(1.0)
        try:
            html = http_get(d, timeout=30).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for href in re.findall(r'href="([^"?]+\.pdf)"', html, re.I):
            found.setdefault(urllib.parse.urljoin(d, href), urllib.parse.unquote(href.split("/")[-1]))
    print(f"  oeb: {len(found)} PDFs from directory listings", flush=True)
    if not found:  # no listing: probe the naming pattern
        for n in range(1, 9):
            for d in (f"g{n}", f"grade{n}"):
                for subj in OEB_SUBJECTS:
                    for kut in ("Kutaa", "KUTAA"):
                        for suf in ("", "_Compressed"):
                            url = f"{OEB_BASE}{d}/{subj}_{kut}-{n}{suf}.pdf"
                            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
                            time.sleep(0.5)
                            try:
                                with urllib.request.urlopen(req, timeout=20) as r:
                                    if r.status == 200:
                                        found.setdefault(url, url.split("/")[-1])
                            except Exception:  # noqa: BLE001
                                pass
        print(f"  oeb: {len(found)} PDFs from name-pattern probing", flush=True)

    def dom(s: str) -> str:
        return "health" if re.search(r"FJQ|fayyaa", s, re.I) else "education"

    yield from pdf_docs(found, work, "textbooks_pdf", "oeb", "C",
                        "(c) Biiroo Barnootaa Oromiyaa; research index only; permission requested", dom)


SOURCES: dict[str, Callable[[Path], Iterator[KBDoc]]] = {
    "wikipedia": src_wikipedia, "tico19": src_tico19, "storybook": src_storybook,
    "ohkb": src_ohkb, "mdh": src_mdh, "textbooks": src_textbooks, "ethnomed": src_ethnomed,
    "webdomain": src_webdomain, "oeb": src_oeb,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="data/kb/raw")
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"
    work.mkdir(exist_ok=True)
    man_path = args.out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name in args.sources.split(","):
        t0 = time.time()
        n = n_ok = words_ok = empty = 0
        tiers: dict[str, int] = defaultdict(int)
        try:
            with gzip.open(args.out / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
                for d in SOURCES[name](work):
                    d = finalize(d)
                    if d.n_words == 0:
                        empty += 1
                        continue
                    n += 1
                    n_ok += d.oromo_ok
                    words_ok += d.n_words if d.oromo_ok else 0
                    tiers[d.tier] += 1
                    fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
            manifest[name] = {"docs": n, "oromo_ok_docs": n_ok, "oromo_ok_words": words_ok, "empty_or_image_only": empty,
                              "tiers": dict(tiers), "seconds": round(time.time() - t0, 1), "status": "ok"}
            print(f"SOURCE {name:<10} docs {n:>6,} | Oromo-check pass {n_ok:>6,} ({(n_ok / n * 100 if n else 0):.0f}%) | "
                  f"words (pass) {words_ok:>10,} | empty/image-only {empty} | tiers {dict(tiers)} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            manifest[name] = {"status": f"failed: {type(exc).__name__}: {str(exc)[:200]}"}
            print(f"SOURCE {name:<10} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    man_path.write_text(json.dumps(manifest, indent=2))
    tot = sum(v.get("oromo_ok_words", 0) for v in manifest.values())
    print(f"KB RAW TOTAL (Oromo-check pass): {tot:,} words across {sum(1 for v in manifest.values() if v.get('status') == 'ok')} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
