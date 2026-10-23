# `gemmas` (Gemma Search)

Fast, on-the-fly semantic code and text search powered by **Google EmbeddingGemma-2** (270M) with Matryoshka 256d/512d representations.

---

## Features
- **Zero-Index Search**: Instantly searches code and documents on the fly without database setup or indexing lag.
- **Dual Modality**:
  - `code` mode: Uses `CodeRetrieval` task prompt for functions, classes, and logic.
  - `text` mode: Uses `Retrieval` task prompt for historical OCR, documentation, notes, and research papers.
- **Ultra-Low Resource Footprint**: Runs in `bfloat16` on CUDA in ~540MB VRAM, or on CPU/MPS with minimal memory.
- **Agent-Ready JSON**: Pass `--json` to pipe structured matches directly to `jq` or autonomous AI agents.
- **Respects `.gitignore`**: Automatically ignores caches, build directories, and virtual environments.

---

## Quick Start

### 1. Direct Execution
```bash
./gemmas/bin/gemmas "where do we discard short pages" run_pilot.py
```

### 2. Using `uv` (Zero-Config Runner)
```bash
# Run directly from repository:
uv run --project gemmas gemmas "search query" .

# Or install globally as a tool:
uv tool install ./gemmas
gemmas "search query" .
```

### 3. Sparse Clone / Standalone Usage
If you sparse checkout or copy only the `gemmas/` directory onto another machine:
```bash
cd gemmas
uv run gemmas "my query" /path/to/project
```

---

## CLI Options

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

Daemon management commands:
  gemmas daemon status  Inspect running daemon, VRAM usage, and idle cooldown countdown
  gemmas daemon stop    Stop daemon immediately and free GPU memory
  gemmas daemon start   Pre-warm daemon and load model in background
  gemmas daemon logs    View daemon log output
```

---

## Persistent LanceDB Vector Indexing

`gemmas` supports fast **persistent vector indexing** powered by **LanceDB**:
- **Sub-15ms Latency**: Queries against an indexed corpus skip raw text chunking and scan directly from disk via zero-copy columnar vectors.
- **Hierarchical Discovery**:
  1. Checks for local in-tree index `<path>/.gemmas.lance/` (travels with dataset).
  2. Checks for central cache `~/.cache/gemmas/indexes/<hash>_<name>.lance/` (zero repository clutter).
  3. Seamlessly falls back to on-the-fly scan if no index exists.
- **Incremental Sync (`--update`)**: Uses file modification timestamps (`mtime`) to only encode newly added or edited files, purging deleted files automatically.

### Indexing Examples:
```bash
# 1. Index corpus into clean central cache (~/.cache/gemmas/indexes/):
./gemmas/bin/gemmas index data/

# 2. Index corpus directly alongside data (data/.gemmas.lance/):
./gemmas/bin/gemmas index data/ --local

# 3. Query automatically uses the index (instant 10ms response):
./gemmas/bin/gemmas "kolera epidemi" data/ -H

# 4. Incrementally update after editing or adding documents:
./gemmas/bin/gemmas index data/ --update

# 5. List all known indexes and disk usage:
./gemmas/bin/gemmas index list

# 6. Bypass index when needed:
./gemmas/bin/gemmas "kolera epidemi" data/ --no-index
```

---

## Examples

### Interactive Search Mode & Inclusion Marking (`-i`)
```bash
./gemmas/bin/gemmas "where do we discard short pages" run_pilot.py -i
```
- **Navigate**: Use `↑`/`↓` (or `j`/`k`) to select a match.
- **Scroll Chunk**: Use `w`/`s` (or `PgUp`/`PgDn`) to scroll long chunks in the detail pane.
- **Mark for Inclusion**: Press `Space`, `Tab`, or `m` to toggle inclusion (`[✓]`). Press `a` to mark all, `c` to clear.
- **Confirm & Output**: Press `Enter` to confirm. Marked file paths are emitted directly to `stdout` (one per line).
- **Edit**: Press `e` or `o` to jump directly into `$EDITOR` at that exact line.
- **Cancel**: Press `q` or `ESC` to abort without outputting paths.

#### Pipe Interactive Selections to Other Tools:
Because `gemmas` decouples the curses terminal (`/dev/tty`) from `stdout`, you can interactively curate files directly within a Unix pipeline:
```bash
# Interactively review and copy chosen documents into a curated folder:
gemmas "kolera epidemi" data/ -i | xargs -I {} cp {} curated_folder/

# Capture interactively reviewed files into a shell variable:
selected=$(gemmas "tuberkulos sanatorium" data/ -i)

# Interactively review and pipe structured JSON (only chosen matches) to jq:
gemmas "kolera epidemi" data/ -i --json | jq '.matches[] | {path: .path, score: .score}'
```

### Search Codebase by Concept
```bash
./gemmas/bin/gemmas "where is 8-bit quantization configured" .
```

### Search Historical Swedish Medical Texts
```bash
./gemmas/bin/gemmas "behandling av kolera eller epidemier" TAXONOMY.py --mode text
```

### Search with Second-Pass Line-Level Heatmap Highlighting (`-H` / `--heat`)
```bash
./gemmas/bin/gemmas "kolera epidemi" data -H
```
- First pass identifies winning 35-line chunk candidates across all files.
- Second pass runs in $< 20\text{ms}$ on GPU across the top matches to compute per-line semantic similarity.
- Top hit lines ($\ge 0.78$) are flagged with `★ score` and highlighted in bold gold/yellow.

### Exclude Previously Reviewed / Blacklisted Files (`--blacklist` / `--black-list`)
Excludes documents by direct filename, list file (`.txt`/`.csv`), or directory tree:
```bash
# Exclude specific files by filename (comma-separated or multiple flags):
./gemmas/bin/gemmas "koppor" data/ --black-list Lakartidningen_1918_0652.txt,Lakartidningen_1918_0655.txt

# Exclude files listed in a text file (one filename per line or CSV):
./gemmas/bin/gemmas "koppor" data/ --black-list excluded_files.txt

# Exclude documents already present in a directory tree:
./gemmas/bin/gemmas "koppor vaccinering" data/ --black-list reviewed_curated/

# Works in both interactive (-i) and pipeline (-l) modes:
./gemmas/bin/gemmas "koppor" data/ --black-list Lakartidningen_1918_0652.txt -l
```

### Pipe JSON Output with Saliency Scores to `jq`
```bash
./gemmas/bin/gemmas "kolera epidemi" data -H --limit 1 --json | jq '.matches[0].highlighted_lines'
```

### Self-Managing Background Daemon & Inactivity Cooldown
By default, `gemmas` automatically launches a background daemon over a local Unix domain socket (`~/.cache/gemmas/daemon.sock`).
- **First Search**: Starts daemon and loads Gemma model into GPU memory (~4.5s).
- **Subsequent Searches**: Run against the warm model in memory (**< 80ms** per search).
- **Autonomous Cooldown**: The daemon monitors its own idle time. If no queries are received within 5 minutes (`--cooldown 300`), it shuts down automatically and frees all VRAM.
- **Inspect Daemon**:
  ```bash
  ./gemmas/bin/gemmas daemon status
  ```
- **Stop Daemon Manually**:
  ```bash
  ./gemmas/bin/gemmas daemon stop
  ```
- **Run Without Daemon**: Pass `--no-daemon` to execute completely in-process.
