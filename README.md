# DataView

A tiny local viewer for your Hugging Face datasets — for when the hosted dataset
viewer isn't available on your plan. Paste a repo id, it downloads into this
folder and shows the rows in a clean table.

Black-and-white, no build step, no external assets — just a small FastAPI
backend and one HTML page.

## Setup

Needs a Python with these packages:

```bash
pip install huggingface_hub datasets fastapi uvicorn pandas pyarrow
```

Log in to Hugging Face once so the viewer can reach your datasets:

```bash
hf auth login
```

## Run

```bash
./run.sh
```

Then open http://127.0.0.1:7860

(Uses your existing `hf` login. Everything — downloads, cache, the library
index — stays inside `./data`, nothing touches `~/.cache`.)

## Use

- **Add** a dataset by repo id (e.g. `owner/dataset-name`).
  If it's not on disk it downloads; if it is, it's shown immediately.
- **Check / refresh** compares your local copy against Hugging Face and pulls
  the update if the commit changed.
- Pick a **table** (each data file in the repo) from the dropdown.
- **Search** scans all columns; **click a row** to see the full record;
  **Stats** shows per-column types, null counts, and value summaries.

## Notes

- Reads `.parquet`, `.jsonl`, `.json`, `.csv`, `.tsv`. Sharded parquet
  (`train-00000-of-00010.parquet`) is grouped into one table.
- Tables are loaded into memory, capped at 100,000 rows. When a table is
  larger, the UI says so and search/stats run on the loaded sample.
- Change the port with `PORT=8000 ./run.sh`.

## Files

- `server.py` — backend (FastAPI): download, update-check, read, paginate, stats.
- `index.html` — the whole UI (no build step, no external assets).
- `data/` — downloaded datasets (`data/repos/<owner>/<name>`) and `library.json`.
