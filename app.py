"""TeraBox data connector for Claude (remote MCP), built for free hosts like Render.

Data flow (nothing needs payment):
  TeraBox --(Colab notebook, once per file)--> Parquet in your PRIVATE Hugging Face dataset
  Claude (any claude.ai chat) --HTTPS + secret key--> this server --> DuckDB on those Parquet files

The server is read-only. It downloads Parquet files from your dataset to its own
temporary disk on first use, runs SQL there, and returns only small results.

Environment (set in the host's encrypted environment settings, never in code):
  ACCESS_KEY   long random string, part of the connector URL (Render generates it)
  HF_TOKEN     Hugging Face READ token (fine-grained, read access to the one dataset)
  HF_DATASET   e.g. "yourname/terabox-data"
  DATA_BASE    optional override of https://huggingface.co/datasets/<HF_DATASET>/resolve/main
  CACHE_DIR    default /tmp/tbcache
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import posixpath
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import duckdb
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse, PlainTextResponse

ACCESS_KEY = os.environ.get("ACCESS_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "").strip().strip("/")
DATA_BASE = (os.environ.get("DATA_BASE")
             or f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main").rstrip("/")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/tmp/tbcache"))
DUCK_TMP = CACHE_DIR / "_duck_tmp"
MEM = os.environ.get("DUCKDB_MEMORY_LIMIT", "256MB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "1"))
QUERY_TIMEOUT = float(os.environ.get("QUERY_TIMEOUT_S", "240"))
MAX_OUTPUT_CHARS = 60_000
for d in (CACHE_DIR, DUCK_TMP):
    d.mkdir(parents=True, exist_ok=True)

_http = httpx.AsyncClient(timeout=httpx.Timeout(60, read=600), follow_redirects=True,
                          headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})
_locks: dict[str, asyncio.Lock] = {}


def norm(p: str) -> str:
    return "/" + posixpath.normpath("/" + (p or "/").strip().lstrip("/")).lstrip("/")


def human(n: float | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{int(n)} B" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return str(n)


def sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


# ------------------------------------------------------------------ dataset access
async def manifest() -> dict:
    r = await _http.get(f"{DATA_BASE}/manifest.json")
    if r.status_code == 404:
        return {"files": {}}
    if r.status_code in (401, 403):
        raise RuntimeError("Hugging Face refused access: check HF_TOKEN and HF_DATASET on the server.")
    r.raise_for_status()
    return r.json()


def local_path(repo_path: str) -> Path:
    return CACHE_DIR / hashlib.sha1(repo_path.encode()).hexdigest()[:16] / posixpath.basename(repo_path)


async def ensure_local(tb_path: str) -> tuple[Path, dict]:
    """Make sure the Parquet for a TeraBox path is on local disk; returns (path, manifest entry)."""
    m = await manifest()
    entry = m.get("files", {}).get(norm(tb_path))
    if not entry:
        raise ValueError(f"{norm(tb_path)} has not been converted yet. Run the Colab notebook for it, "
                         "then call list_datasets to see what is available.")
    repo_path = entry["parquet"]
    dest = local_path(repo_path)
    lock = _locks.setdefault(repo_path, asyncio.Lock())
    async with lock:
        if dest.exists() and dest.stat().st_size == entry.get("parquet_size", dest.stat().st_size):
            return dest, entry
        need = int(entry.get("parquet_size") or 0)
        free = shutil.disk_usage(CACHE_DIR).free
        if need and need > free - 64 * 1024**2:
            raise RuntimeError(f"Not enough temporary disk on the server for {human(need)} "
                               f"(free {human(free)}). Ask Claude to call clear_cache first.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        async with _http.stream("GET", f"{DATA_BASE}/{repo_path}") as r:
            if r.status_code != 200:
                raise RuntimeError(f"download of {repo_path} failed: HTTP {r.status_code}")
            with open(part, "wb") as fh:
                async for c in r.aiter_bytes(1 << 20):
                    fh.write(c)
        part.replace(dest)
    return dest, entry


TB_REF = re.compile(r"""tb\(\s*(['"])(.+?)\1\s*\)""")


async def resolve_sql(sql: str) -> str:
    out, pos = [], 0
    for m in TB_REF.finditer(sql):
        path, _ = await ensure_local(m.group(2))
        out.append(sql[pos:m.start()] + f"read_parquet({sql_lit(str(path))})")
        pos = m.end()
    out.append(sql[pos:])
    return "".join(out)


def run_query(sql: str, max_rows: int) -> dict:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM}'")
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET temp_directory={sql_lit(str(DUCK_TMP))}")
    con.execute(f"SET allowed_directories=[{sql_lit(str(CACHE_DIR) + '/')}]")
    con.execute("SET enable_external_access=false")  # no other files, no network, no extensions
    con.execute("SET lock_configuration=true")
    timer = threading.Timer(QUERY_TIMEOUT, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        if cur.description is None:
            return {"columns": [], "rows": [], "truncated": False}
        rows = cur.fetchmany(max_rows + 1)
        return {"columns": [d[0] for d in cur.description], "rows": rows[:max_rows],
                "truncated": len(rows) > max_rows}
    finally:
        timer.cancel()
        con.close()


def fmt(res: dict) -> str:
    if not res["columns"]:
        return "OK (no rows)."

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.10g}"
        return str(v).replace("\n", " ").replace(",", ";")

    text = "\n".join([",".join(res["columns"])] + [",".join(cell(v) for v in r) for r in res["rows"]])
    note = ""
    if len(text) > MAX_OUTPUT_CHARS:
        text, note = text[:MAX_OUTPUT_CHARS], "\n...[cut at 60k characters; aggregate more]"
    if res["truncated"]:
        note += f"\n...[more rows exist; first {len(res['rows'])} shown. Aggregate or raise max_rows]"
    return text + note


# ------------------------------------------------------------------ MCP tools
mcp = FastMCP(
    "terabox-data",
    instructions=(
        "Read-only access to the user's TeraBox data, converted to Parquet and stored in their private "
        "Hugging Face dataset. Use list_datasets first, then describe_data, then query_data with DuckDB SQL "
        "where tb('/original/TeraBox/path.csv') is the table. Return aggregates, never dump raw rows. "
        "The server is small (0.1 CPU, 512 MB RAM): prefer narrow column selections and GROUP BY. "
        "The first call after idle can take about a minute while the server wakes up."
    ),
    stateless_http=True,
    json_response=True,
    log_level="WARNING",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def list_datasets() -> str:
    """Files that have been converted from TeraBox and are ready to query."""
    m = await manifest()
    files = m.get("files", {})
    if not files:
        return "No converted files yet. Run the Colab notebook (TeraBox_to_HuggingFace.ipynb) first."
    lines = [f"{len(files)} file(s):"]
    for p, e in sorted(files.items()):
        lines.append(f"{p}  rows={e.get('rows'):,}  source={human(e.get('source_size'))}  "
                     f"parquet={human(e.get('parquet_size'))}  converted={e.get('converted_at', '')[:10]}")
    return "\n".join(lines)


@mcp.tool()
async def describe_data(path: str) -> str:
    """Columns, types, row count and first 5 rows of a converted file (use its TeraBox path)."""
    ref = f"tb({sql_lit(norm(path))})"
    schema = await asyncio.to_thread(run_query, await resolve_sql(f"DESCRIBE SELECT * FROM {ref}"), 500)
    count = await asyncio.to_thread(run_query, await resolve_sql(f"SELECT count(*) FROM {ref}"), 1)
    head = await asyncio.to_thread(run_query, await resolve_sql(f"SELECT * FROM {ref} LIMIT 5"), 5)
    s = {"columns": ["column", "type"], "rows": [r[:2] for r in schema["rows"]], "truncated": False}
    return f"rows: {count['rows'][0][0]:,}\n\nschema:\n{fmt(s)}\n\nfirst rows:\n{fmt(head)}"


@mcp.tool()
async def query_data(sql: str, max_rows: int = 200) -> str:
    """DuckDB SQL over converted files; refer to a file as tb('/TeraBox/path/file.csv').
    Example daily bars from 1-minute data:
      SELECT date_trunc('day', ts) d, arg_min(open, ts) o, max(high) h, min(low) l,
             arg_max(close, ts) c, sum(volume) v FROM tb('/Futures/ES/es_1min.csv') GROUP BY 1 ORDER BY 1
    Output is capped at max_rows (<=5000) and 60k characters."""
    sql2 = await resolve_sql(sql)
    return fmt(await asyncio.to_thread(run_query, sql2, min(max(max_rows, 1), 5000)))


@mcp.tool()
async def server_status() -> dict:
    """Temporary disk use on the server and which files are already loaded."""
    du = shutil.disk_usage(CACHE_DIR)
    loaded = [p.name for p in CACHE_DIR.glob("*/*") if p.is_file() and not p.name.endswith(".part")]
    return {"disk_free": human(du.free), "disk_total": human(du.total), "loaded_files": loaded,
            "memory_limit": MEM, "note": "Loaded files are lost when the free server sleeps; they reload automatically."}


@mcp.tool()
async def clear_cache() -> str:
    """Delete locally loaded Parquet copies (never touches TeraBox or Hugging Face)."""
    for p in CACHE_DIR.iterdir():
        if p != DUCK_TMP:
            shutil.rmtree(p, ignore_errors=True)
    return "Server cache cleared."


# ------------------------------------------------------------------ HTTP layer
mcp_app = mcp.streamable_http_app()


async def app(scope, receive, send):
    """/ -> health; /c/<ACCESS_KEY>/mcp -> MCP (claude.ai); /mcp + Bearer <ACCESS_KEY> -> MCP."""
    if scope["type"] == "lifespan":
        await mcp_app(scope, receive, send)
        return
    if scope["type"] != "http":
        return
    path: str = scope["path"]
    if path in ("/", "/healthz"):
        await PlainTextResponse("ok")(scope, receive, send)
        return
    if len(ACCESS_KEY) < 24:
        await PlainTextResponse("server misconfigured: ACCESS_KEY missing or too short",
                                status_code=503)(scope, receive, send)
        return
    ok = False
    if path.startswith("/c/"):
        key, _, tail = path[3:].partition("/")
        if hmac.compare_digest(key.encode(), ACCESS_KEY.encode()):
            ok = True
            scope = dict(scope, path="/" + tail, raw_path=("/" + tail).encode())
    elif path.startswith("/mcp"):
        auth = dict((k.decode().lower(), v.decode()) for k, v in scope.get("headers", [])).get("authorization", "")
        ok = auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), ACCESS_KEY.encode())
    if not ok:
        await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
        return
    if scope["path"].rstrip("/") == "/mcp":
        scope = dict(scope, path="/mcp", raw_path=b"/mcp")
    await mcp_app(scope, receive, send)
