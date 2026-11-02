"""HTTP API and web chat for GadaaBot 2.0.

  POST /api/ask   {"question": "..."} -> {"answer", "sources", "abstained", "emergency", "seconds"}
  POST /api/feedback {"question", "answer", "helpful": bool, "comment"}  (appended to a JSONL file)
  GET  /healthz
  GET  /          minimal mobile-first chat page (no external assets, works on slow connections)

Run (after the vLLM server and Qdrant are up):
  GADAABOT_ROOT=~/gadaabot uvicorn gadaabot.serve.api:app --host 0.0.0.0 --port 8080
Environment: GADAABOT_LLM_URL, GADAABOT_QDRANT_URL, GADAABOT_ROOT, GADAABOT_RERANKER, GADAABOT_FEEDBACK.
Questions are not stored unless the user sends feedback.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gadaabot.serve.pipeline import GadaaBot, Settings

ROOT = Path(os.environ.get("GADAABOT_ROOT", ".")).expanduser()
FEEDBACK = Path(os.environ.get("GADAABOT_FEEDBACK", str(ROOT / "logs" / "feedback.jsonl")))
RATE = int(os.environ.get("GADAABOT_RATE_PER_MIN", "12"))

app = FastAPI(title="GadaaBot 2.0", version="2.0")
_bot: GadaaBot | None = None
_hits: dict[str, deque] = defaultdict(deque)


def bot() -> GadaaBot:
    global _bot  # noqa: PLW0603
    if _bot is None:
        _bot = GadaaBot(Settings(
            llm_url=os.environ.get("GADAABOT_LLM_URL", "http://127.0.0.1:48000"),
            qdrant_url=os.environ.get("GADAABOT_QDRANT_URL", "http://127.0.0.1:46333"),
            afrie5=ROOT / "models/McGill-NLP/AfriE5-Large-instruct",
            reranker=Path(os.environ.get("GADAABOT_RERANKER", str(ROOT / "models/BAAI/bge-reranker-v2-m3"))),
            parents_db=ROOT / "data/kb/index/parents.sqlite"))
    return _bot


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Feedback(BaseModel):
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=6000)
    helpful: bool
    comment: str = Field(default="", max_length=1000)


def _limit(ip: str) -> None:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Too many questions; try again in a minute.")
    q.append(now)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(body: Ask, request: Request) -> dict:
    _limit(request.client.host if request.client else "?")
    r = bot().ask(body.question)
    return {"answer": r.answer, "sources": r.sources, "abstained": r.abstained, "emergency": r.emergency,
            "seconds": r.seconds}


@app.post("/api/feedback")
def feedback(body: Feedback) -> dict[str, str]:
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": int(time.time()), **body.model_dump()}, ensure_ascii=False) + "\n")
    return {"status": "saved"}


PAGE = """<!doctype html><html lang="om"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GadaaBot</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--card:#fff;--line:#e4e3df;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--card:#232322;--line:#383835;--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}h1{font-size:20px;margin:8px 0}p.s{color:var(--muted);margin:0 0 12px}
#log{display:flex;flex-direction:column;gap:10px;margin-bottom:96px}.m{padding:10px 12px;border-radius:10px;white-space:pre-wrap}
.q{align-self:flex-end;background:var(--acc);color:#fff;max-width:85%}.a{background:var(--card);border:1px solid var(--line)}
form{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px}
.row{max-width:720px;margin:0 auto;display:flex;gap:8px}textarea{flex:1;font:inherit;padding:8px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--ink);resize:none;height:48px}
button{font:inherit;padding:0 16px;border:0;border-radius:8px;background:var(--acc);color:#fff}
.fb{margin-top:6px;font-size:14px}.fb button{background:transparent;color:var(--muted);border:1px solid var(--line);padding:2px 8px}
</style></head><body><main>
<h1>GadaaBot</h1><p class="s">Gaaffii fayyaa fi barnootaa Afaan Oromootiin gaafadhu. Deebiin maddeewwan isaa waliin kennama.
Odeeffannoon fayyaa gorsa ogeessa fayyaa bakka hin bu'u.</p><div id="log"></div></main>
<form id="f"><div class="row"><textarea id="t" placeholder="Gaaffii kee asitti barreessi..." required></textarea>
<button>Ergi</button></div></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),t=document.getElementById('t');
function add(cls,text){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;log.appendChild(d);
window.scrollTo(0,document.body.scrollHeight);return d}
f.onsubmit=async e=>{e.preventDefault();const q=t.value.trim();if(!q)return;t.value='';add('q',q);
const w=add('a','...');try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:q})});const j=await r.json();w.textContent=r.ok?j.answer:(j.detail||'Dogoggora');
if(r.ok){const fb=document.createElement('div');fb.className='fb';for(const [lab,h] of [['Gargaareera',true],['Hin gargaarre',false]]){
const b=document.createElement('button');b.type='button';b.textContent=lab;b.onclick=()=>{fetch('/api/feedback',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,answer:j.answer,helpful:h})});fb.textContent='Galatoomi!'};
fb.appendChild(b);fb.append(' ')}w.appendChild(fb)}}catch(err){w.textContent='Dogoggora: '+err}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE
