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
  -e, --ext EXT         Comma-separated extensions (e.g. 'py,ts,md')
  --mode {auto,code,text}
                        Search mode (default: auto)
  --dim {256,512,768}   Matryoshka representation dimension (default: 256)
  --chunk-lines N       Chunk window size in lines (default: 35)
  --overlap N           Overlap between chunks in lines (default: 10)
  --json                Output structured JSON
  -v, --version         Show version
```

---

## Examples

### Search Codebase by Concept
```bash
./gemmas/bin/gemmas "where is 8-bit quantization configured" .
```

### Search Historical Swedish Medical Texts
```bash
./gemmas/bin/gemmas "behandling av kolera eller epidemier" TAXONOMY.py --mode text
```

### Pipe JSON Output to `jq`
```bash
./gemmas/bin/gemmas "GPU cache clearing" run_pilot.py --json | jq '.matches[0]'
```
