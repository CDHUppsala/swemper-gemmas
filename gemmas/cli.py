import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import argparse
from pathlib import Path
from typing import List

from . import __version__
from .chunker import collect_files, chunk_file, infer_mode
from .searcher import GemmaSearchEngine
from .formatter import format_terminal_output, format_json_output
from .interactive import run_interactive_browser
from .daemon import (
    ensure_daemon,
    is_daemon_alive,
    daemon_rpc,
    daemon_search_rpc,
    get_socket_path,
    get_log_path,
)

def handle_daemon_command(cmd_args: List[str]) -> int:
    cmd = cmd_args[0] if cmd_args else "status"
    sock_path = get_socket_path()
    log_path = get_log_path()

    if cmd == "status":
        resp = daemon_rpc({"action": "status"})
        if resp and resp.get("status") == "running":
            vram = resp.get("vram", {})
            vram_str = f"{vram.get('allocated_mb', 0):.1f} MB" if "allocated_mb" in vram else "N/A"
            dev_name = vram.get("device_name", resp.get("device", "auto"))
            idle = resp.get("idle_seconds", 0)
            cooldown = resp.get("cooldown", 300)
            remaining = max(0, cooldown - idle)
            rem_m, rem_s = divmod(int(remaining), 60)

            print(f"Gemmas Daemon: RUNNING (PID {resp.get('pid')})")
            print(f"  Model:           {resp.get('model')}")
            print(f"  Device:          {resp.get('device')} ({dev_name})")
            print(f"  VRAM Allocated:  {vram_str}")
            print(f"  Uptime:          {resp.get('uptime_seconds', 0):.1f}s")
            print(f"  Total Requests:  {resp.get('total_requests', 0)}")
            print(f"  Idle Timer:      {idle:.1f}s / {cooldown}s (unloads in {rem_m}m {rem_s}s)")
            print(f"  Socket:          {sock_path}")
            return 0
        else:
            print(f"Gemmas Daemon: NOT RUNNING")
            print(f"  Socket:          {sock_path} (inactive)")
            return 0

    elif cmd == "stop":
        resp = daemon_rpc({"action": "stop"})
        if resp and resp.get("status") in ("stopping", "ok"):
            print("Gemmas Daemon: Stopped successfully.")
        else:
            if sock_path.exists():
                try:
                    sock_path.unlink()
                except OSError:
                    pass
            print("Gemmas Daemon: Not running (or already stopped).")
        return 0

    elif cmd == "start":
        if is_daemon_alive(sock_path):
            print("Gemmas Daemon: Already running.")
            return 0
        ok = ensure_daemon(verbose=True)
        return 0 if ok else 1

    elif cmd == "logs":
        if log_path.exists():
            lines = log_path.read_text().splitlines()[-30:]
            print("\n".join(lines))
        else:
            print("No daemon logs found.")
        return 0

    else:
        print(f"Unknown daemon command: '{cmd}'. Available: status, stop, start, logs")
        return 1

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemmas",
        description="Fast on-the-fly semantic code and text search powered by Google EmbeddingGemma-2.",
        epilog=(
            "Daemon management:\n"
            "  gemmas daemon status   Check daemon status, idle cooldown, and VRAM\n"
            "  gemmas daemon stop     Stop daemon immediately and free GPU memory\n"
            "  gemmas daemon start    Pre-warm daemon and load model in background\n"
            "  gemmas daemon logs     View recent daemon logs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "query",
        type=str,
        help="Semantic search query (e.g. 'where do we handle retries', 'kolera epidemi')"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=".",
        help="Target directory or single file to search (default: current directory)"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=5,
        help="Maximum number of matches to return (default: 5)"
    )
    parser.add_argument(
        "-e", "--ext",
        type=str,
        default=None,
        help="Comma-separated file extensions to include (e.g. 'py,ts,md')"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "code", "text"],
        default="auto",
        help="Search mode: 'code' (CodeRetrieval prompt), 'text' (Retrieval prompt), or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[256, 512, 768],
        default=256,
        help="Matryoshka representation dimension (default: 256 for maximum speed)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sentences", "lines", "auto"],
        default="sentences",
        help="Chunking strategy: 'sentences' (paragraph-aligned sentence accumulation, default), 'lines' (fixed line window), or 'auto' (sentences for text, lines for code)"
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=5,
        help="Minimum complete sentences per chunk for sentence strategy (default: 5)"
    )
    parser.add_argument(
        "--sentence-overlap",
        type=int,
        default=3,
        help="Sentence overlap between consecutive chunks for sentence strategy (default: 3)"
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=35,
        help="Sliding chunk window size in lines for lines strategy (default: 35)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="Overlap between consecutive chunks in lines for lines strategy (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64; reduce if VRAM is constrained)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length in tokens to bound memory (default: 512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default: auto)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/embeddinggemma-2",
        help="HuggingFace model identifier or local path (default: google/embeddinggemma-2)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive terminal browser to navigate matches and inspect full chunks with arrow keys"
    )
    parser.add_argument(
        "-H", "--heat", "--highlight",
        dest="heat",
        action="store_true",
        help="Run second-pass line-level semantic heatmap to highlight and colorize top matching lines"
    )
    parser.add_argument(
        "--heat-threshold",
        type=float,
        default=0.70,
        help="Minimum similarity threshold for line-level highlighting (default: 0.70)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=300,
        help="Daemon inactivity cooldown in seconds before unloading model from VRAM (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Bypass background daemon and run model directly in-process"
    )
    parser.add_argument(
        "--dehyphen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic soft dehyphenation of line-broken words in text mode (default: enabled; disable with --no-dehyphen)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as structured JSON (useful for piping to jq or agents)"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"gemmas {__version__}"
    )

    return parser.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "daemon":
        return handle_daemon_command(raw_args[1:])

    args = parse_args(argv)
    target_path = Path(args.path)

    if not target_path.exists():
        sys.stderr.write(f"Error: Target path '{args.path}' does not exist.\n")
        return 1

    ext_list = [e.strip() for e in args.ext.split(",")] if args.ext else None

    start_total = time.perf_counter()

    # 1. Collect files
    files = collect_files(target_path, extensions=ext_list)
    if not files:
        if args.json:
            sys.stdout.write(format_json_output([], args.query, 0, 0, 0.0, args.mode, args.dim) + "\n")
        else:
            sys.stderr.write(f"No matching text files found in '{args.path}'.\n")
        return 0

    # 2. Chunk files
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(
            f,
            strategy=args.strategy,
            chunk_lines=args.chunk_lines,
            overlap=args.overlap,
            min_sentences=args.min_sentences,
            sentence_overlap=args.sentence_overlap
        ))

    if not all_chunks:
        if args.json:
            sys.stdout.write(format_json_output([], args.query, 0, len(files), 0.0, args.mode, args.dim) + "\n")
        else:
            sys.stderr.write(f"No non-empty chunks found across {len(files)} files.\n")
        return 0

    # 3. Determine search mode
    search_mode = args.mode
    if search_mode == "auto":
        search_mode = infer_mode(files)

    show_feedback = not args.json

    if show_feedback:
        sys.stderr.write(f"[*] Scanned {len(files)} files -> {len(all_chunks)} candidate chunks\n")

    # 4. Search Execution (Daemon IPC or in-process)
    results = None
    if not args.no_daemon:
        # Check or launch daemon
        ready = ensure_daemon(
            cooldown=args.cooldown,
            model_name=args.model,
            device=args.device,
            dim=args.dim,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_len,
            verbose=show_feedback
        )
        if ready:
            resp = daemon_search_rpc({
                "action": "search",
                "query": args.query,
                "chunks": all_chunks,
                "mode": search_mode,
                "limit": args.limit,
                "dim": args.dim,
                "heat": args.heat,
                "dehyphen": args.dehyphen
            }, timeout=180.0, show_feedback=show_feedback)
            if resp and resp.get("status") == "ok":
                results = []
                for m in resp.get("matches", []):
                    c = m["chunk"]
                    if "line_scores" in c and isinstance(c["line_scores"], dict):
                        c["line_scores"] = {
                            int(k): float(v) for k, v in c["line_scores"].items() if str(k).isdigit()
                        }
                    results.append((c, m["score"]))
            else:
                if show_feedback:
                    sys.stderr.write("[-] Daemon RPC returned error, falling back to local engine.\n")

    if results is None:
        # Local in-process fallback
        engine = GemmaSearchEngine(
            model_name=args.model,
            device=args.device,
            dim=args.dim,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_len
        )
        results = engine.search(
            query=args.query,
            chunks=all_chunks,
            mode=search_mode,
            limit=args.limit,
            dehyphen=args.dehyphen,
            show_progress=(show_feedback and len(all_chunks) > 50)
        )
        if args.heat and results:
            if show_feedback:
                sys.stderr.write("[*] Pass 2: Computing line-level saliency heatmaps...\n")
            engine.compute_line_saliency(
                query=args.query,
                chunks=[r[0] for r in results],
                mode=search_mode,
                dehyphen=args.dehyphen
            )

    elapsed_ms = (time.perf_counter() - start_total) * 1000

    # 5. Output
    if args.interactive:
        if show_feedback:
            sys.stderr.write("[*] Launching interactive browser...\n")
        run_interactive_browser(
            results=results,
            query=args.query,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold
        )
    elif args.json:
        sys.stdout.write(format_json_output(
            results=results,
            query=args.query,
            total_chunks=len(all_chunks),
            file_count=len(files),
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold
        ) + "\n")
    else:
        sys.stdout.write(format_terminal_output(
            results=results,
            query=args.query,
            total_chunks=len(all_chunks),
            file_count=len(files),
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold
        ) + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
