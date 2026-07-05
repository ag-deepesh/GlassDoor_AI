# GlassBox — local setup (Mac + Windows)

Two servers run locally: the Python/FastAPI backend (`api/`, `core/`) and the
Vite/React frontend (`frontend/`). Nothing needs to be deployed anywhere —
everything runs on your machine, and API keys are typed into the running UI
(or passed as CLI flags/env vars), never written to a file.

## A note if this repo lives in a synced folder (OneDrive/Dropbox/iCloud)

This project was built inside an OneDrive-synced folder. That's fine for the
code itself, but two things are worth knowing:

- **Knowledge Bases** (`data/kbs/<name>/`) contain a Chroma/SQLite database.
  Cloud-sync tools can lock or partially-sync a database file mid-write,
  which can corrupt it. If you hit strange Chroma errors after switching
  machines, check whether OneDrive created a conflict-copy file inside
  `data/kbs/<name>/chroma/` (usually named like `chroma.sqlite3 (conflicted
  copy ...)`) — delete the conflict copy, or just pause OneDrive sync while
  you're actively building/querying a KB.
- If you'd rather sidestep this entirely, set `GLASSBOX_DATA_DIR` (see
  below) to a path outside the synced folder — the code and git history
  stay in the synced repo, only the KB data (and optionally your venv)
  move out.

## Prerequisites

| | macOS | Windows |
|---|---|---|
| Python | 3.11 (`brew install python@3.11`, or [python.org](https://python.org)) | 3.11 from [python.org](https://python.org) — check "Add python.exe to PATH" during install |
| Node.js | 18+ (`brew install node`, or [nodejs.org](https://nodejs.org)) | 18+ from [nodejs.org](https://nodejs.org) |
| Tesseract (OCR) | `brew install tesseract` | Installer at https://github.com/UB-Mannheim/tesseract/wiki, then add its install dir (e.g. `C:\Program Files\Tesseract-OCR`) to PATH |
| Git | `brew install git` or Xcode CLT | [git-scm.com](https://git-scm.com) |

The pipeline needs real Python **3.11+** (not the OS-bundled `python3`, which
on macOS is often 3.9 and too old for this codebase's type-hint syntax).
Check with `python3.11 --version` — if that command doesn't exist, install
Python 3.11 first.

## Backend

```bash
# macOS
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

That's it — one command. (Earlier Milestone 2 notes warned that `ragas` and
`langchain-anthropic` need to be installed in a specific order to avoid a
dependency conflict; `requirements.txt` now pins `ragas==0.4.3` and
`core/evaluation/ragas_eval.py` shims one import ragas still makes into a
submodule `langchain-community` has since dropped, so a single
`pip install -r requirements.txt` resolves cleanly.)

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or
`http://127.0.0.1:8000/registry` to confirm every stage's registered methods
loaded correctly.

### Optional: point KB storage/venv outside a synced folder

```bash
# macOS/Linux
export GLASSBOX_DATA_DIR="$HOME/glassbox-data"
# Windows (PowerShell)
$env:GLASSBOX_DATA_DIR = "$HOME\glassbox-data"
```

Set this before running `uvicorn` (and re-set it each new terminal session,
or add it to your shell profile). If unset, KB data lives in `./data`
inside the repo.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. It expects the backend at
`http://127.0.0.1:8000` by default — override with a `.env` file in
`frontend/` containing `VITE_API_BASE=http://127.0.0.1:8000` if you run the
API on a different port/host.

## Using it

1. Open the UI, click **+ New KB**, name it, attach your documents (PDF/
   DOCX/PPTX/MD/TXT — ≤20 files, ≤25MB/file, ≤150MB total), click **Build
   KB**. Parsing → chunking → embedding stream in as they complete.
2. Once built, the KB is selectable from the dropdown any time after —
   embedding only ever runs once per corpus.
3. Type a query, optionally toggle **web search** and the **ReAct loop**,
   click **Run**. Add API keys via the **🔑 API keys** button in the top bar
   first — generation, non-local embeddings, the ReAct judge, and web search
   all need a real key for their provider.
4. **Tavily** (web search) needs its own key from https://tavily.com (has a
   free tier) — without one, the web toggle degrades gracefully to KB-only
   results with a note in the trace, it won't hard-fail the run.

## CLI (no UI needed)

```bash
python -m cli.main show-registry
python -m cli.main run sample_corpus -q "How does self-attention work?" --claude-key sk-ant-...
python -m cli.main run sample_corpus -q "..." --claude-key sk-ant-... --step   # pause after each stage
python -m cli.main run sample_corpus -q "..." --claude-key sk-ant-... --web --tavily-key tvly-... --react
```

## Docker (later — cloud deploy path, not needed for local dev)

```bash
docker compose up --build
```

Builds the backend (FastAPI + Chroma, `data/` as a mounted volume so KBs
persist across container restarts) and the frontend (static Vite build
served via nginx). API keys are still typed into the running UI, or passed
as environment variables to the backend container at deploy time — never
baked into the image. See `Dockerfile.backend`, `Dockerfile.frontend`,
`docker-compose.yml`.

## Troubleshooting

- **"attempt to write a readonly database"** — a stale Chroma/SQLite handle
  from a crashed process, or (if this repo is in a synced folder) a sync
  conflict. Restart the backend; if it persists, check for OneDrive
  conflict-copy files under `data/kbs/<name>/chroma/`.
- **Embedding hangs the first time** — MiniLM/BGE-small download from
  huggingface.co on first use (a few hundred MB); subsequent runs are fast
  since the model is cached locally (`~/.cache/huggingface` by default).
- **`ModuleNotFoundError` for something in `langchain_community`** — make
  sure you installed via `requirements.txt` (which pins `ragas==0.4.3`), not
  a different `ragas` version; newer `ragas` releases may drift further
  from what `core/evaluation/ragas_eval.py`'s shim covers.
