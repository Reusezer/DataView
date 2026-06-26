"""DataView — a minimal local viewer for Hugging Face datasets.

No frills: paste a repo id, it downloads into ./data, and shows the rows.
Everything (downloads, cache, library index) stays inside this project folder.
"""
import io
import os
import re
import json
import math
import threading
from pathlib import Path

# Read the HF token from its real location BEFORE we pin HF_HOME below,
# otherwise the lib looks for the token inside ./data and finds nothing.
from huggingface_hub import get_token, HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPOS = DATA / "repos"
LIBRARY_FILE = DATA / "library.json"
TOKEN = get_token()

# Keep all HF scratch files inside the project, not ~/.cache.
os.environ.setdefault("HF_HOME", str(DATA / ".hf"))
REPOS.mkdir(parents=True, exist_ok=True)

api = HfApi(token=TOKEN)

# The private data-vault clone (managed by the data-vault skill). Browsable
# here but not part of this repo — its location is local config, never committed.
VAULT_DIR = Path(os.environ.get("DATA_VAULT_DIR", str(Path.home() / "data-vault")))

# How many rows we load into memory per table. Honestly labelled in the UI
# when a table is larger than this.
MAX_ROWS = 100_000
DATA_EXTS = {".parquet", ".jsonl", ".json", ".csv", ".tsv"}

# ---------------------------------------------------------------------------
# Library index (which datasets the user has added)
# ---------------------------------------------------------------------------
_lib_lock = threading.Lock()


def load_library():
    if LIBRARY_FILE.exists():
        return json.loads(LIBRARY_FILE.read_text())
    return []


def save_library(lib):
    LIBRARY_FILE.write_text(json.dumps(lib, indent=2))


def lib_entry(lib, repo_id):
    return next((e for e in lib if e["repo_id"] == repo_id), None)


def repo_dir(repo_id):
    return REPOS / repo_id


# ---------------------------------------------------------------------------
# Table loading (in-memory cache keyed by repo_id + table key)
# ---------------------------------------------------------------------------
_table_cache = {}
_cache_lock = threading.Lock()

SHARD_RE = re.compile(r"^(.*)-\d{4,5}-of-\d{4,5}\.parquet$")


def _logical_ext(p):
    """Extension a file should be read as, ignoring a trailing .gz and
    treating .ndjson as .jsonl. So 'train.jsonl.gz' -> '.jsonl'."""
    name = p.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    ext = p.suffix.lower() if "." not in name else "." + name.rsplit(".", 1)[-1]
    return ".jsonl" if ext == ".ndjson" else ext


def _open_text(path):
    """Open a possibly-gzipped file as UTF-8 text."""
    import gzip
    sp = str(path)
    if sp.lower().endswith(".gz"):
        return gzip.open(sp, "rt", encoding="utf-8")
    return open(sp, encoding="utf-8")


def discover_tables(repo_id):
    """Return a list of logical tables for a downloaded repo.

    Each table is one data file, except sharded parquet (train-00000-of-00010)
    which is grouped back into a single logical table.
    """
    base = repo_dir(repo_id)
    groups = {}  # key -> {"key", "label", "ext", "paths"}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(base).parts):
            continue  # skip .cache / .git etc
        ext = _logical_ext(p)
        if ext not in DATA_EXTS:
            continue
        rel = p.relative_to(base).as_posix()
        m = SHARD_RE.match(p.name)
        if ext == ".parquet" and m:
            stem = m.group(1)
            key = (p.parent.relative_to(base) / stem).as_posix()
            label = key
        else:
            key = rel
            label = rel
        g = groups.setdefault(key, {"key": key, "label": label, "ext": ext, "paths": []})
        g["paths"].append(p)
    # stable order, paths sorted for shard concatenation
    tables = []
    for g in sorted(groups.values(), key=lambda x: x["key"]):
        g["paths"] = sorted(g["paths"])
        tables.append(g)
    return tables


# Keys that conventionally hold the list of records inside a wrapper object.
RECORD_KEYS = ("data", "rows", "records", "examples", "items", "samples",
               "instances", "results", "queries", "entries")


def _extract_records(data):
    """Turn a parsed .json value into something pandas.DataFrame reads as rows.

    Auto-adjusts to the shapes JSON datasets show up in:
      - a plain list of records                     -> used as-is
      - a wrapper object {"metadata":..., "queries":[...]} -> the record list
      - a columnar dict {"q": [...], "a": [...]}     -> kept (columns)
      - a lone list under any key                    -> one column "value"
      - anything else                                -> a single row
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return [data]

    def is_records(v):
        return isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)

    record_lists = {k: v for k, v in data.items() if is_records(v)}
    if record_lists:
        for k in RECORD_KEYS:                       # prefer conventional names
            if k in record_lists:
                return record_lists[k]
        return max(record_lists.values(), key=len)  # else the longest list

    list_vals = {k: v for k, v in data.items() if isinstance(v, list)}
    # columnar dict: every value is a same-length list -> these are columns
    if list_vals and len(list_vals) == len(data):
        lengths = {len(v) for v in list_vals.values()}
        if len(lengths) == 1:
            return data
    if len(list_vals) == 1:                         # a lone list of scalars
        return [{"value": x} for x in next(iter(list_vals.values()))]
    return [data]


def _read_dataframe(table):
    """Load a table (possibly sharded) into a pandas DataFrame, capped at
    MAX_ROWS. Returns (df, true_total, truncated)."""
    import pandas as pd

    ext = table["ext"]
    paths = table["paths"]

    if ext == ".parquet":
        import pyarrow.parquet as pq

        true_total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        frames, got = [], 0
        for p in paths:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=8192):
                frames.append(batch.to_pandas())
                got += batch.num_rows
                if got >= MAX_ROWS:
                    break
            if got >= MAX_ROWS:
                break
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df) > MAX_ROWS:
            df = df.iloc[:MAX_ROWS]
        return df, true_total, true_total > len(df)

    if ext == ".jsonl":
        rows, true_total = [], 0
        for p in paths:
            with _open_text(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    true_total += 1
                    if len(rows) < MAX_ROWS:
                        rows.append(json.loads(line))
        return pd.DataFrame(rows), true_total, true_total > len(rows)

    if ext == ".json":
        with _open_text(paths[0]) as f:
            data = json.load(f)
        data = _extract_records(data)
        df = pd.DataFrame(data)
        true_total = len(df)
        if len(df) > MAX_ROWS:
            df = df.iloc[:MAX_ROWS]
        return df, true_total, true_total > len(df)

    # csv / tsv
    sep = "\t" if ext == ".tsv" else ","
    frames = [pd.read_csv(p, sep=sep, nrows=MAX_ROWS + 1) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    true_total = len(df)
    truncated = len(df) > MAX_ROWS
    if truncated:
        df = df.iloc[:MAX_ROWS]
    return df, true_total, truncated


def get_table(repo_id, table_key):
    ck = (repo_id, table_key)
    with _cache_lock:
        if ck in _table_cache:
            return _table_cache[ck]
    tables = discover_tables(repo_id)
    table = next((t for t in tables if t["key"] == table_key), None)
    if table is None:
        raise KeyError(table_key)
    df, true_total, truncated = _read_dataframe(table)
    result = {"df": df, "true_total": true_total, "truncated": truncated}
    with _cache_lock:
        _table_cache[ck] = result
    return result


def clear_cache(repo_id):
    with _cache_lock:
        for k in [k for k in _table_cache if k[0] == repo_id]:
            del _table_cache[k]


def get_file_table(abs_path):
    """Load a single data file (used by the vault browser) as a cached table."""
    abs_path = Path(abs_path)
    ck = ("__file__", str(abs_path))
    with _cache_lock:
        if ck in _table_cache:
            return _table_cache[ck]
    table = {"ext": _logical_ext(abs_path), "paths": [abs_path]}
    df, true_total, truncated = _read_dataframe(table)
    result = {"df": df, "true_total": true_total, "truncated": truncated}
    with _cache_lock:
        _table_cache[ck] = result
    return result


# ---------------------------------------------------------------------------
# Cell / value serialization for JSON responses
# ---------------------------------------------------------------------------
def to_jsonable(v):
    import numpy as np
    import pandas as pd

    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [to_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


# ---------------------------------------------------------------------------
# HTTP API (FastAPI)
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="DataView")
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def remote_sha(repo_id):
    try:
        return api.dataset_info(repo_id).sha
    except Exception:
        return None


@app.get("/api/library")
def get_library():
    lib = load_library()
    out = []
    for e in lib:
        downloaded = repo_dir(e["repo_id"]).exists()
        out.append({
            "repo_id": e["repo_id"],
            "added_at": e.get("added_at"),
            "local_sha": e.get("local_sha"),
            "downloaded": downloaded,
            "desc": e.get("desc", ""),
        })
    return out


class DescReq(BaseModel):
    desc: str


@app.post("/api/library/{repo_id:path}/describe")
def describe_dataset(repo_id: str, req: DescReq):
    with _lib_lock:
        lib = load_library()
        entry = lib_entry(lib, repo_id)
        if entry is None:
            raise HTTPException(404, "Not in library.")
        entry["desc"] = req.desc.strip()
        save_library(lib)
    return {"repo_id": repo_id, "desc": req.desc.strip()}


class AddReq(BaseModel):
    repo_id: str


@app.post("/api/library")
def add_dataset(req: AddReq):
    repo_id = req.repo_id.strip().strip("/")
    if not repo_id or repo_id.count("/") != 1:
        raise HTTPException(400, "Expected a repo id like 'owner/name'.")
    try:
        info = api.dataset_info(repo_id)
    except Exception as e:
        raise HTTPException(404, f"Could not find dataset '{repo_id}': {e}")

    target = repo_dir(repo_id)
    snapshot_download(
        repo_id, repo_type="dataset", local_dir=str(target),
        token=TOKEN, cache_dir=str(DATA / ".hf" / "cache"),
    )
    with _lib_lock:
        lib = load_library()
        entry = lib_entry(lib, repo_id)
        if entry is None:
            entry = {"repo_id": repo_id}
            lib.append(entry)
        entry["local_sha"] = info.sha
        entry["added_at"] = entry.get("added_at") or _now()
        save_library(lib)
    clear_cache(repo_id)
    return {"repo_id": repo_id, "local_sha": info.sha}


@app.post("/api/library/{repo_id:path}/refresh")
def refresh_dataset(repo_id: str):
    with _lib_lock:
        lib = load_library()
        if lib_entry(lib, repo_id) is None:
            raise HTTPException(404, "Not in library.")
    info = api.dataset_info(repo_id)
    snapshot_download(
        repo_id, repo_type="dataset", local_dir=str(repo_dir(repo_id)),
        token=TOKEN, cache_dir=str(DATA / ".hf" / "cache"),
    )
    with _lib_lock:
        lib = load_library()
        lib_entry(lib, repo_id)["local_sha"] = info.sha
        save_library(lib)
    clear_cache(repo_id)
    return {"repo_id": repo_id, "local_sha": info.sha}


@app.get("/api/library/{repo_id:path}/check")
def check_update(repo_id: str):
    with _lib_lock:
        lib = load_library()
        entry = lib_entry(lib, repo_id)
    if entry is None:
        raise HTTPException(404, "Not in library.")
    sha = remote_sha(repo_id)
    return {
        "repo_id": repo_id,
        "local_sha": entry.get("local_sha"),
        "remote_sha": sha,
        "update_available": bool(sha) and sha != entry.get("local_sha"),
    }


@app.delete("/api/library/{repo_id:path}")
def remove_dataset(repo_id: str, delete_files: bool = Query(False)):
    with _lib_lock:
        lib = load_library()
        if lib_entry(lib, repo_id) is None:
            raise HTTPException(404, "Not in library.")
        lib = [e for e in lib if e["repo_id"] != repo_id]
        save_library(lib)
    clear_cache(repo_id)
    if delete_files:
        import shutil
        d = repo_dir(repo_id)
        if d.exists():
            shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/dataset/{repo_id:path}/tables")
def list_tables(repo_id: str):
    if not repo_dir(repo_id).exists():
        raise HTTPException(404, "Not downloaded.")
    tables = discover_tables(repo_id)
    if not tables:
        raise HTTPException(404, "No readable data files (.parquet/.jsonl/.json/.csv/.tsv) found.")
    out = []
    for t in tables:
        data = get_table(repo_id, t["key"])
        df = data["df"]
        out.append({
            "key": t["key"],
            "label": t["label"],
            "ext": t["ext"],
            "columns": list(df.columns),
            "rows_loaded": int(len(df)),
            "true_total": int(data["true_total"]),
            "truncated": data["truncated"],
        })
    return out


def build_rows(data, offset, limit, q):
    df = data["df"]
    if q:
        ql = q.lower()
        mask = df.apply(
            lambda col: col.map(lambda v: ql in str(v).lower()), axis=0
        ).any(axis=1)
        df = df[mask]
    total = int(len(df))
    page = df.iloc[offset:offset + limit]
    rows = [[to_jsonable(v) for v in rec] for rec in page.to_numpy()]
    return {
        "columns": list(df.columns),
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "matched": total,
        "true_total": int(data["true_total"]),
        "truncated": data["truncated"],
        "filtered": bool(q),
    }


def build_stats(data):
    import pandas as pd

    df = data["df"]
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        non_null = int(s.notna().sum())
        info = {
            "name": c,
            "dtype": str(s.dtype),
            "non_null": non_null,
            "nulls": int(n - non_null),
        }
        try:
            if pd.api.types.is_numeric_dtype(s):
                d = s.dropna()
                if len(d):
                    info["min"] = to_jsonable(d.min())
                    info["max"] = to_jsonable(d.max())
                    info["mean"] = round(float(d.mean()), 4)
            else:
                hashable = s.dropna().map(
                    lambda v: v if isinstance(v, (str, int, float, bool)) else json.dumps(to_jsonable(v))
                )
                info["unique"] = int(hashable.nunique())
                top = hashable.value_counts().head(5)
                info["top"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
        except Exception:
            pass
        cols.append(info)
    return {"rows": int(n), "true_total": int(data["true_total"]),
            "truncated": data["truncated"], "columns": cols}


def build_jsonl(data):
    """The loaded table as JSONL text — one JSON object per row."""
    df = data["df"]
    cols = list(df.columns)
    lines = [json.dumps({c: to_jsonable(v) for c, v in zip(cols, rec)}, ensure_ascii=False)
             for rec in df.to_numpy()]
    return "\n".join(lines) + ("\n" if lines else "")


def _jsonl_response(data, filename):
    return Response(
        build_jsonl(data), media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/dataset/{repo_id:path}/rows")
def get_rows(repo_id: str, table: str, offset: int = 0, limit: int = 50, q: str = ""):
    try:
        data = get_table(repo_id, table)
    except KeyError:
        raise HTTPException(404, "Unknown table.")
    return JSONResponse(build_rows(data, offset, limit, q))


@app.get("/api/dataset/{repo_id:path}/download")
def download_hf(repo_id: str, table: str):
    try:
        data = get_table(repo_id, table)
    except KeyError:
        raise HTTPException(404, "Unknown table.")
    name = f"{repo_id.split('/')[-1]}__{Path(table).stem}.jsonl"
    return _jsonl_response(data, name)


@app.get("/api/dataset/{repo_id:path}/stats")
def get_stats(repo_id: str, table: str):
    try:
        data = get_table(repo_id, table)
    except KeyError:
        raise HTTPException(404, "Unknown table.")
    return build_stats(data)


# ---------------------------------------------------------------------------
# Vault — browse a private GitHub repo of datasets (the data-vault skill).
# Structure: <category>/<project>/<dataset> with a manifest.json index.
# ---------------------------------------------------------------------------
def _vault_resolve(rel):
    """Resolve a vault-relative path and refuse anything outside the vault."""
    p = (VAULT_DIR / rel).resolve()
    if not str(p).startswith(str(VAULT_DIR.resolve())):
        raise HTTPException(400, "Bad path.")
    return p


@app.get("/api/vault/tree")
def vault_tree():
    manifest = VAULT_DIR / "manifest.json"
    if not manifest.exists():
        return {"configured": False, "dir": str(VAULT_DIR), "categories": []}
    m = json.loads(manifest.read_text())
    cats = []
    for cat, cdata in sorted(m.get("categories", {}).items()):
        projects = []
        for proj, pdata in sorted(cdata.get("projects", {}).items()):
            datasets = []
            for name, ds in sorted(pdata.get("datasets", {}).items()):
                present = (VAULT_DIR / ds["file"]).exists()
                datasets.append({
                    "name": name, "file": ds["file"], "format": ds.get("format"),
                    "rows": ds.get("rows"), "bytes": ds.get("bytes"),
                    "desc": ds.get("desc", ""), "source": ds.get("source", ""),
                    "lfs": ds.get("lfs", False), "present": present,
                })
            projects.append({"name": proj, "datasets": datasets})
        cats.append({"name": cat, "projects": projects})
    return {"configured": True, "dir": str(VAULT_DIR),
            "updated": m.get("updated"), "categories": cats}


@app.get("/api/vault/rows")
def vault_rows(path: str, offset: int = 0, limit: int = 50, q: str = ""):
    p = _vault_resolve(path)
    if not p.exists():
        raise HTTPException(404, "File not present locally — fetch it first.")
    return JSONResponse(build_rows(get_file_table(p), offset, limit, q))


@app.get("/api/vault/stats")
def vault_stats(path: str):
    p = _vault_resolve(path)
    if not p.exists():
        raise HTTPException(404, "File not present locally — fetch it first.")
    return build_stats(get_file_table(p))


@app.get("/api/vault/download")
def vault_download(path: str):
    p = _vault_resolve(path)
    if not p.exists():
        raise HTTPException(404, "File not present locally — fetch it first.")
    return _jsonl_response(get_file_table(p), f"{p.stem}.jsonl")


class VaultDescReq(BaseModel):
    path: str
    desc: str


@app.post("/api/vault/describe")
def vault_describe(req: VaultDescReq):
    import subprocess
    manifest_path = VAULT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Vault not configured.")
    m = json.loads(manifest_path.read_text())
    found = None
    for cat in m.get("categories", {}).values():
        for proj in cat.get("projects", {}).values():
            for ds in proj.get("datasets", {}).values():
                if ds.get("file") == req.path:
                    found = ds
    if found is None:
        raise HTTPException(404, "Dataset not in manifest.")
    found["desc"] = req.desc.strip()
    manifest_path.write_text(json.dumps(m, indent=2) + "\n")
    # commit + best-effort push so the description syncs to the repo
    subprocess.run(["git", "-C", str(VAULT_DIR), "add", "manifest.json"],
                   text=True, capture_output=True)
    subprocess.run(["git", "-C", str(VAULT_DIR), "commit", "-q", "-m",
                    f"Describe {req.path}"], text=True, capture_output=True)
    push = subprocess.run(["git", "-C", str(VAULT_DIR), "push", "-q"],
                          text=True, capture_output=True)
    return {"ok": True, "path": req.path, "desc": req.desc.strip(),
            "synced": push.returncode == 0}


class FetchReq(BaseModel):
    path: str


@app.post("/api/vault/fetch")
def vault_fetch(req: FetchReq):
    import subprocess
    _vault_resolve(req.path)  # validate
    # restore from git (handles both regular checkout and LFS smudge)
    r = subprocess.run(["git", "-C", str(VAULT_DIR), "checkout", "--", req.path],
                       text=True, capture_output=True)
    subprocess.run(["git", "-C", str(VAULT_DIR), "lfs", "pull", "--include", req.path],
                   text=True, capture_output=True)
    if not _vault_resolve(req.path).exists():
        raise HTTPException(500, f"Could not fetch {req.path}: {r.stderr.strip()}")
    return {"ok": True, "path": req.path}


def _now():
    # avoid Date import games; good enough for "added_at" display
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"\n  DataView  →  http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
