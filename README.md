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
- **Search** scans all columns; type **`#42`** to jump to a row by its index.
  **Click a row** to see the full record; **Stats** shows per-column types,
  null counts, and value summaries.

### Look up a row from the terminal

The `#N` index (shown in the viewer's `#` column, 0-based) can be fetched
headlessly — handy for scripts and agents:

```bash
./run.sh row <path> <N> [count]      # prints the row(s) as JSON
# <path> is absolute or vault-relative; row 0 is the first record
```

## Vault (optional)

Beyond Hugging Face, DataView can browse a **private GitHub repo you use as your
own dataset store** — handy for keeping project data organized in one place
without making it public. Point it at a local clone:

```bash
DATA_VAULT_DIR=~/data-vault ./run.sh      # default is ~/data-vault
```

It expects the repo to be organized by **category / project / dataset** with a
`manifest.json` index at the root:

```
<vault>/
  manifest.json
  <category>/<project>/<dataset>.jsonl
```

`manifest.json` schema:

```json
{
  "version": 1,
  "updated": "2026-06-25",
  "categories": {
    "research": {
      "projects": {
        "my-project": {
          "datasets": {
            "examples": {
              "file": "research/my-project/examples.jsonl",
              "format": "jsonl", "rows": 320, "bytes": 51234,
              "lfs": false, "source": "", "desc": "short description",
              "added": "2026-06-25"
            }
          }
        }
      }
    }
  }
}
```

The **Vault** section in the sidebar lists everything by category → project →
dataset; click to view. Datasets not present locally (e.g. after a light clone,
or Git LFS files) show a **fetch** action that restores them from the remote.
The vault location is read from `DATA_VAULT_DIR` only — nothing about your
private repo is stored in this project.

## Notes

- Reads `.parquet`, `.jsonl`, `.json` (incl. nested record lists and columnar
  dicts), `.csv`, `.tsv`, and gzipped variants. Sharded parquet
  (`train-00000-of-00010.parquet`) is grouped into one table.
- Tables are loaded into memory, capped at 100,000 rows. When a table is
  larger, the UI says so and search/stats run on the loaded sample.
- Change the port with `PORT=8000 ./run.sh`.

## Files

- `server.py` — backend (FastAPI): download, update-check, read, paginate, stats.
- `index.html` — the whole UI (no build step, no external assets).
- `data/` — downloaded datasets (`data/repos/<owner>/<name>`) and `library.json`.
