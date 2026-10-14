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
  --cooldown N          Daemon inactivity cooldown in seconds (default: 300 = 5 min)
  --no-daemon           Bypass background daemon and run model directly in-process
  --mode {auto,code,text}
                        Search mode (default: auto)
  --dim {256,512,768}   Matryoshka representation dimension (default: 256)
  --chunk-lines N       Chunk window size in lines (default: 35)
  --overlap N           Overlap between chunks in lines (default: 10)
  --batch-size N        Batch size for inference (default: 32)
  --max-seq-len N       Max token length to bound memory (default: 512)
  --json                Output structured JSON
  -v, --version         Show version

Daemon management commands:
  gemmas daemon status  Inspect running daemon, VRAM usage, and idle cooldown countdown
  gemmas daemon stop    Stop daemon immediately and free GPU memory
  gemmas daemon start   Pre-warm daemon and load model in background
  gemmas daemon logs    View daemon log output
```

---

## Examples

### Interactive Search Mode (Browse with Arrow Keys)
```bash
./gemmas/bin/gemmas "where do we discard short pages" run_pilot.py -i
```
- Use `↑`/`↓` (or `j`/`k`) to select a match.
- The right/bottom pane automatically expands the **full chunk with line numbers**.
- Use `w`/`s` (or `PgUp`/`PgDn`) to scroll long chunks.
- Press `o` or `Enter` to open the file directly in `$EDITOR` at that exact line!
- Press `q` or `ESC` to exit back to the shell.

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
