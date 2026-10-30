# `swemper-gemmas` (`gemmas`)

Fast, on-the-fly and indexed semantic code and text search CLI powered by **Google EmbeddingGemma-2** (270M) with Matryoshka 256d/512d representations and **LanceDB** vector storage.

---

## Origins & Research Context

`gemmas` was originally developed as an experimental semantic search, retrieval, and corpus curation tool within the **SweMPer** research project (*Swedish Medical Periodicals*, 1781–2014, spanning 633,000 pages across 22 historical periodicals). 

In digital humanities research and NLP workflows over historical OCR text, conventional keyword grep struggles with archaic orthography, optical character errors, and line-broken compound words (*samman-sättningar*). `gemmas` was designed to solve this with:
- Zero-lag zero-index on-the-fly scanning for instant exploration.
- Persistent sub-15ms vector indexing with LanceDB for large corpora.
- Soft dehyphenation preserving Swedish compound words across line breaks.
- Interactive terminal review and inclusion curation (`-i`) that pipes selected files directly into Unix workflows (`xargs`, `cp`, `jq`).

While developed in SweMPer, `gemmas` is a fully standalone, general-purpose CLI for semantic search across both **codebases** (Python, Rust, C++, Go, etc.) and **prose corpora** (research notes, historical archives, documentation).

---

## Features

- **Dual Search Modes**:
  - **Zero-Index On-The-Fly**: Scan directories or single files instantly without database setup or indexing lag.
  - **Persistent LanceDB Indexing**: Sub-15ms vector searches over tens of thousands of documents with incremental timestamp-based updates (`--update`).
- **Dual Modality Task Prompts**:
  - `code` mode: Uses EmbeddingGemma `CodeRetrieval` task prompt for function definitions, classes, and logic.
  - `text` mode: Uses `Retrieval` task prompt for prose, historical documents, research notes, and papers.
- **Interactive Review & Pipeline Curation (`-i`)**:
  - Split-view curses TUI to navigate matches and inspect full chunk contexts.
  - Keystroke inclusion marking (`Space`/`Tab`) to select files for downstream processing.
  - Unix pipe ready: Selected paths are output directly to `stdout` upon pressing `Enter`.
- **Soft Dehyphenation**:
  - Automatically reconstructs compound words split across line breaks (common in printed periodicals and OCR text).
- **Second-Pass Semantic Heatmap (`-H` / `--heat`)**:
  - Fast line-level second pass highlights key lines with scores in bold gold/yellow.
- **Autonomous GPU Daemon**:
  - Keeps the EmbeddingGemma model resident in GPU memory (~540MB in `bfloat16`) for sub-80ms queries.
  - Shuts down automatically after 5 minutes of inactivity (`--cooldown 300`) to free all VRAM.
- **Pipeline & Agent Ready**:
  - Clean non-interactive flags: `-l` (`--files`), `--compact`, `--plain`, `-q` (`--quiet`), and `--json`.
  - Exclude already curated documents with `--blacklist` / `--black-list` / `--blacklist-dir`.

---

## Installation & Setup

### 1. Using `uv` (Recommended)
```bash
# Install globally as a CLI tool:
uv tool install git+https://github.com/mattslindstrom/swemper-gemmas.git

# Or install editable from a local clone:
git clone https://github.com/mattslindstrom/swemper-gemmas.git
cd swemper-gemmas
uv tool install -e .
```

### 2. Standard `pip` Installation
```bash
pip install git+https://github.com/mattslindstrom/swemper-gemmas.git

# Or for local development:
pip install -e .
```

### 3. Direct Execution via Wrapper Script
If using the repository directly without installing globally:
```bash
./bin/gemmas "search query" /path/to/files
```

---

## Hardware & Memory Guidance

- **Model Footprint**: The EmbeddingGemma-2 270M model runs in `bfloat16` and requires only **~540MB VRAM** on CUDA / Apple Silicon (MPS), or minimal RAM on CPU.
- **Indexing Large Corpora**:
  - Default inference batch size is `64`.
  - On consumer GPUs with **8GB–12GB VRAM** (e.g. RTX 3080/4070), we recommend passing `--batch-size 32` when indexing large text collections to avoid memory spikes from 512-token attention activations:
    ```bash
    gemmas index my_corpus/ --batch-size 32
    ```
- **Inactivity Cooldown**: The background daemon monitors its idle time and automatically terminates after 5 minutes, completely freeing GPU memory.

---

## Quick Reference

### Command Line Synopsis
```text
usage: gemmas [-h] [-n LIMIT] [-e EXT] [--mode {auto,code,text}]
              [--dim {256,512,768}] [--chunk-lines CHUNK_LINES]
              [--overlap OVERLAP] [--device DEVICE] [--model MODEL]
              [--json] [-v]
              query [path]

positional arguments:
  query                 Semantic search query
  path                  Target directory or file (default: .)

options:
  -h, --help            Show help message
  -n, --limit LIMIT     Max matches to return (default: 5)
  -i, --interactive     Interactive split-view TUI to browse and inspect full chunks
  -H, --heat            Run second-pass line-level semantic heatmap to highlight key lines
  --heat-threshold N    Threshold for line highlighting (default: 0.70)
  --strategy {sentences,lines,auto}
                        Chunking strategy (default: sentences)
  --min-sentences N     Minimum sentences per chunk in sentence strategy (default: 5)
  --sentence-overlap N  Sentence overlap between consecutive chunks (default: 3)
  --chunk-lines N       Chunk window size in lines for lines strategy (default: 35)
  --overlap N           Overlap between chunks in lines for lines strategy (default: 3)
  --cooldown N          Daemon inactivity cooldown in seconds (default: 300 = 5 min)
  --no-daemon           Bypass background daemon and run model directly in-process
  --dehyphen, --no-dehyphen
                        Enable/disable soft dehyphenation of line-broken compound words (default: enabled)
  --mode {auto,code,text}
                        Search mode (default: auto)
  --dim {256,512,768}   Matryoshka representation dimension (default: 256)
  --batch-size N        Batch size for inference (default: 64)
  --max-seq-len N       Max token length to bound memory (default: 512)
  --index-dir PATH      Explicit directory containing a LanceDB index to query
  --no-index            Bypass existing LanceDB index and force on-the-fly chunk scanning
  --blacklist, --black-list, --blacklist-dir PATH
                        Exclude hits matching filenames, list files (.txt/.csv), or directories
  -l, --files           Print only the paths of matching files (one per line, ideal for pipes and xargs)
  --compact, --oneline  Print each match on a single line (score, path:line, brief snippet)
  --plain, --no-header  Omit search banner and execution footer for clean scripting output
  -q, --quiet           Suppress all output; exit with code 0 if matches found, 1 otherwise
  --json                Output structured JSON
  -v, --version         Show version

Persistent Index Management (LanceDB):
  gemmas index <path>          Index target directory into central cache (~/.cache/gemmas/indexes/)
  gemmas index <path> --local  Index target directory alongside data (<path>/.gemmas.lance/)
  gemmas index <path> --update Incrementally sync modified/added files
  gemmas index <path> --info   Inspect index statistics and disk footprint
  gemmas index list            List all known indexes in cache
  gemmas index <path> --drop   Delete index

Daemon Management Commands:
  gemmas daemon status         Inspect running daemon, VRAM usage, and idle cooldown countdown
  gemmas daemon stop           Stop daemon immediately and free GPU memory
  gemmas daemon start          Pre-warm daemon and load model in background
  gemmas daemon logs           View daemon log output
```

---

## Practical Examples

### 1. Codebase Search
```bash
# Locate functional logic across a codebase:
gemmas "where do we validate authentication tokens" src/

# Search specific file types:
gemmas "retry logic with exponential backoff" . -e .py,.rs

# Inspect candidate code chunks interactively:
gemmas "database connection pool error handling" src/ -i
```

### 2. Historical Texts & Digital Humanities Search
```bash
# Query historical Swedish medical periodicals with line heatmap:
gemmas "behandling av kolera eller epidemier" corpus/ --mode text -H

# Search with soft dehyphenation for broken compounds:
gemmas "tuberkulos sanatorievård" corpus/ --dehyphen

# Filter out previously reviewed documents:
gemmas "smittkoppor vaccination" corpus/ --black-list excluded_files.txt
```

### 3. Interactive Curation & Unix Pipeline Integration
`gemmas` cleanly separates curses terminal drawing (`/dev/tty`) from file output (`stdout`). This allows interactive curation directly in Unix pipes:
```bash
# Browse search hits interactively; mark candidate documents with Space; press Enter:
gemmas "kolera epidemier" corpus/ -i | xargs -I {} cp {} curated_samples/

# Capture interactively reviewed files into a shell variable:
reviewed=$(gemmas "sanatorievård" corpus/ -i)

# Pipe only marked items as structured JSON into jq:
gemmas "kolera" corpus/ -i --json | jq '.matches[] | {path: .path, score: .score}'
```

### 4. Persistent LanceDB Vector Indexing
```bash
# 1. Build a persistent index in central cache (~/.cache/gemmas/indexes/):
gemmas index corpus/

# 2. Or index alongside the dataset directory (corpus/.gemmas.lance/):
gemmas index corpus/ --local

# 3. Subsequent queries execute in < 15ms directly from vector index:
gemmas "kolera epidemi" corpus/ -H

# 4. Incrementally update only modified or added documents:
gemmas index corpus/ --update

# 5. List all active indexes:
gemmas index list
```

### 5. Self-Managing Background Daemon
```bash
# Check daemon memory and idle cooldown timer:
gemmas daemon status

# Stop daemon manually:
gemmas daemon stop

# Run completely in-process without daemon:
gemmas "query" . --no-daemon
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Developed by Matts Lindström as part of the SweMPer research project.
