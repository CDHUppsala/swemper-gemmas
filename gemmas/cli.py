import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import argparse
from pathlib import Path
from typing import List

from . import __version__
from .chunker import collect_files, chunk_file, infer_mode, collect_blacklist_filenames
from .searcher import GemmaSearchEngine
from .interactive import run_interactive_browser
from .formatter import (
    format_terminal_output,
    format_json_output,
    format_compact_output,
    format_files_only_output,
)
from .daemon import (
    ensure_daemon,
    is_daemon_alive,
    daemon_rpc,
    daemon_search_rpc,
    daemon_encode_query_rpc,
    daemon_line_saliency_rpc,
    get_socket_path,
    get_log_path,
)
from .index_store import (
    LanceIndexManager,
    resolve_index_path,
    get_cache_index_path,
    get_local_index_path,
    list_all_indexes,
)
from .indexer import build_index

def handle_index_command(cmd_args: List[str]) -> int:
    if cmd_args and cmd_args[0] == "list":
        indexes = list_all_indexes()
        if not indexes:
            print("No cached LanceDB indexes found in ~/.cache/gemmas/indexes/")
            return 0
        print(f"{'Source Root':<45} {'Files':<8} {'Chunks':<9} {'Size (MB)':<10} {'Index Path'}")
        print("=" * 110)
        for idx in indexes:
            src = idx.get("source_root", "unknown")
            files = str(idx.get("total_files", 0))
            chunks = str(idx.get("total_chunks", 0))
            size = f"{idx.get('disk_size_mb', 0):.2f}"
            path_str = idx.get("index_path", "")
            print(f"{src:<45} {files:<8} {chunks:<9} {size:<10} {path_str}")
        return 0

    parser = argparse.ArgumentParser(
        prog="gemmas index",
        description="Build, update, or manage persistent LanceDB vector indexes for gemmas.",
        epilog=(
            "Examples:\n"
            "  gemmas index data/                     # Index into central cache (~/.cache/gemmas/indexes/)\n"
            "  gemmas index data/ --local             # Index into local data/.gemmas.lance/\n"
            "  gemmas index data/ --update            # Incrementally sync modified/added files\n"
            "  gemmas index data/ --info              # Show metadata and statistics for target index\n"
            "  gemmas index data/ --drop              # Delete target index\n"
            "  gemmas index list                      # List all known cached indexes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=str, nargs="?", default=".", help="Target directory to index (default: current directory)")
    parser.add_argument("--local", action="store_true", help="Store index in <path>/.gemmas.lance (alongside data)")
    parser.add_argument("--index-dir", type=str, default=None, help="Explicit destination directory for Lance index")
    parser.add_argument("--update", action="store_true", help="Incremental update: only re-index new or modified files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index and rebuild from scratch")
    parser.add_argument("--info", action="store_true", help="Show index metadata and statistics")
    parser.add_argument("--drop", action="store_true", help="Delete the index")
    parser.add_argument("-e", "--ext", type=str, default=None, help="Comma-separated file extensions to include")
    parser.add_argument("--strategy", choices=["sentences", "lines", "auto"], default="sentences")
    parser.add_argument("--min-sentences", type=int, default=5)
    parser.add_argument("--sentence-overlap", type=int, default=3)
    parser.add_argument("--chunk-lines", type=int, default=35)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--dim", type=int, choices=[256, 512, 768], default=256)
    parser.add_argument("--dehyphen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--model", type=str, default="google/embeddinggemma-2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cooldown", type=int, default=300)
    parser.add_argument("--no-daemon", action="store_true")
    parser.add_argument(
        "--blacklist", "--black-list", "--blacklist-dir",
        "--exclude-dir-files", "--exclude-from-dir", "--exclude-files",
        dest="blacklist_items",
        action="append",
        type=str,
        default=[],
        help="Exclude files matching filenames, list files, or directories from being indexed"
    )

    args = parser.parse_args(cmd_args)
    target_p = Path(args.path).resolve()

    if args.info or args.drop:
        idx_p = resolve_index_path(target_p, explicit_dir=args.index_dir)
        if not idx_p or not idx_p.exists():
            print(f"No existing index found for '{args.path}'.")
            return 1
        mgr = LanceIndexManager(idx_p)
        if args.drop:
            mgr.drop()
            print(f"Deleted index at '{idx_p}'.")
            return 0
        if args.info:
            meta = mgr.get_metadata()
            total_bytes = sum(f.stat().st_size for f in idx_p.rglob("*") if f.is_file())
            mb = round(total_bytes / (1024 * 1024), 2)
            print(f"Index Location:    {idx_p}")
            print(f"Source Root:       {meta.get('source_root', str(target_p))}")
            print(f"Model:             {meta.get('model_name', 'google/embeddinggemma-2')}")
            print(f"Dimension:         {meta.get('dim', 256)}d")
            print(f"Strategy:          {meta.get('strategy', 'sentences')}")
            print(f"Dehyphenation:     {meta.get('dehyphen', True)}")
            print(f"Indexed Chunks:    {meta.get('total_chunks', len(mgr.table) if mgr.table else 0)}")
            print(f"Indexed Files:     {meta.get('total_files', len(mgr.get_indexed_files()))}")
            print(f"Disk Size:         {mb} MB")
            indexed_at = meta.get("indexed_at")
            if indexed_at:
                print(f"Last Indexed:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(indexed_at))}")
            return 0

    blacklist_filenames = set()
    if args.blacklist_items:
        items_to_scan = []
        for entry in args.blacklist_items:
            for part in entry.replace(",", " ").split():
                part = part.strip()
                if part:
                    items_to_scan.append(part)
        blacklist_filenames = collect_blacklist_filenames(items_to_scan)

    ext_list = [e.strip() for e in args.ext.split(",")] if args.ext else None
    res_dest = build_index(
        target_path=target_p,
        index_dir=args.index_dir,
        is_local=args.local,
        update=args.update,
        overwrite=args.overwrite,
        extensions=ext_list,
        strategy=args.strategy,
        min_sentences=args.min_sentences,
        sentence_overlap=args.sentence_overlap,
        chunk_lines=args.chunk_lines,
        overlap=args.overlap,
        dim=args.dim,
        dehyphen=args.dehyphen,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        model_name=args.model,
        device=args.device,
        cooldown=args.cooldown,
        no_daemon=args.no_daemon,
        show_feedback=True,
        blacklist_filenames=blacklist_filenames
    )
    return 0 if res_dest is not None else 1


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
            "Index management:\n"
            "  gemmas index <path>          Index target directory into central cache (~/.cache/gemmas/indexes/)\n"
            "  gemmas index <path> --local  Index target directory alongside data (<path>/.gemmas.lance/)\n"
            "  gemmas index <path> --update Incrementally sync modified/added files\n"
            "  gemmas index <path> --info   Inspect index statistics and disk footprint\n"
            "  gemmas index list            List all known indexes in cache\n\n"
            "Daemon management:\n"
            "  gemmas daemon status         Check daemon status, idle cooldown, and VRAM\n"
            "  gemmas daemon stop           Stop daemon immediately and free GPU memory\n"
            "  gemmas daemon start          Pre-warm daemon and load model in background\n"
            "  gemmas daemon logs           View recent daemon logs\n"
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
        "--index-dir",
        type=str,
        default=None,
        help="Explicit directory containing a LanceDB index to query"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Bypass any existing LanceDB index and force on-the-fly chunk scanning"
    )
    parser.add_argument(
        "-l", "--files-with-matches", "--files",
        dest="files_with_matches",
        action="store_true",
        help="Print only the paths of matching files (one per line, ideal for pipes and xargs)"
    )
    parser.add_argument(
        "--compact", "--oneline",
        dest="compact",
        action="store_true",
        help="Print each match on a single line (score, path:line, brief snippet)"
    )
    parser.add_argument(
        "--plain", "--no-header",
        dest="plain",
        action="store_true",
        help="Omit search banner and execution footer for clean scripting output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all standard output; exit with code 0 if matches found, 1 otherwise"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as structured JSON (useful for piping to jq or agents)"
    )
    parser.add_argument(
        "--blacklist", "--black-list", "--blacklist-dir",
        "--exclude-dir-files", "--exclude-from-dir", "--exclude-files",
        dest="blacklist_items",
        action="append",
        type=str,
        default=[],
        help="Exclude hits matching filenames, list files (.txt/.csv), or recursively scanned directories"
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
    if raw_args and raw_args[0] == "index":
        return handle_index_command(raw_args[1:])

    args = parse_args(argv)
    target_path = Path(args.path).resolve()

    if not target_path.exists():
        sys.stderr.write(f"Error: Target path '{args.path}' does not exist.\n")
        return 1

    start_total = time.perf_counter()
    show_feedback = not (args.json or args.files_with_matches or args.quiet or args.plain)

    # 0. Resolve blacklist items if specified (directories, list files, or filenames)
    blacklist_filenames: Set[str] = set()
    if args.blacklist_items:
        items_to_scan = []
        for entry in args.blacklist_items:
            for part in entry.replace(",", " ").split():
                part = part.strip()
                if part:
                    items_to_scan.append(part)
        blacklist_filenames = collect_blacklist_filenames(items_to_scan)
        if show_feedback and blacklist_filenames:
            sys.stderr.write(
                f"[*] Loaded blacklist of {len(blacklist_filenames)} filenames to exclude.\n"
            )

    # 1. Check for persistent LanceDB index (Hierarchical Discovery)
    idx_path = None
    if not args.no_index:
        idx_path = resolve_index_path(target_path, explicit_dir=args.index_dir)

    results = None
    total_chunks_count = 0
    file_count = 0
    search_mode = args.mode

    if idx_path is not None:
        manager = LanceIndexManager(idx_path)
        meta = manager.get_metadata()
        if search_mode == "auto":
            search_mode = meta.get("mode", "text")
        dim = args.dim if args.dim != 256 else meta.get("dim", 256)
        source_root = Path(meta.get("source_root", target_path))

        if show_feedback:
            loc_label = "in-tree local" if ".gemmas.lance" in str(idx_path) else "cached"
            sys.stderr.write(f"[*] Using {loc_label} LanceDB index at '{idx_path}'\n")

        # Encode query
        q_vec = None
        if not args.no_daemon:
            ready = ensure_daemon(
                cooldown=args.cooldown,
                model_name=args.model,
                device=args.device,
                dim=dim,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_len,
                verbose=False
            )
            if ready:
                q_vec = daemon_encode_query_rpc(args.query, mode=search_mode, dim=dim)

        if q_vec is None:
            engine = GemmaSearchEngine(
                model_name=args.model,
                device=args.device,
                dim=dim,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_len
            )
            q_vec = engine.encode_query(args.query, mode=search_mode)

        # Search Lance table (with dynamic over-fetch to satisfy limit when blacklisting)
        raw_matches = []
        if manager.table is not None and len(manager.table) > 0:
            fetch_limit = args.limit
            if blacklist_filenames:
                multiplier = 3 if len(blacklist_filenames) < 20 else 5
                fetch_limit = min(len(manager.table), max(args.limit * multiplier, args.limit + 50))

            while True:
                candidate_matches = manager.search(q_vec, limit=fetch_limit)
                raw_matches = []
                for chunk, score in candidate_matches:
                    rel_p = chunk.get("path", "")
                    disp_path = str(source_root / rel_p)
                    chunk["path"] = disp_path

                    if blacklist_filenames:
                        fname = Path(disp_path).name
                        if fname in blacklist_filenames or rel_p in blacklist_filenames:
                            continue

                    raw_matches.append((chunk, score))
                    if len(raw_matches) >= args.limit:
                        break

                if len(raw_matches) >= args.limit or fetch_limit >= len(manager.table) or not candidate_matches:
                    break

                next_limit = min(len(manager.table), fetch_limit * 3)
                if next_limit <= fetch_limit:
                    break
                fetch_limit = next_limit

        results = raw_matches[:args.limit]

        # Pass 2: Heatmap
        if args.heat and results and not (args.files_with_matches or args.quiet):
            if show_feedback:
                sys.stderr.write("[*] Pass 2: Computing line-level saliency heatmaps...\n")
            top_chunks = [r[0] for r in results]
            heat_applied = False
            if not args.no_daemon and is_daemon_alive(get_socket_path()):
                heat_chunks = daemon_line_saliency_rpc(
                    query=args.query,
                    chunks=top_chunks,
                    mode=search_mode,
                    dehyphen=args.dehyphen
                )
                if heat_chunks:
                    for orig, hc in zip(top_chunks, heat_chunks):
                        orig["line_scores"] = {
                            int(k): float(v) for k, v in hc.get("line_scores", {}).items() if str(k).isdigit()
                        }
                    heat_applied = True

            if not heat_applied:
                engine = GemmaSearchEngine(
                    model_name=args.model,
                    device=args.device,
                    dim=dim,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_len
                )
                engine.compute_line_saliency(
                    query=args.query,
                    chunks=top_chunks,
                    mode=search_mode,
                    dehyphen=args.dehyphen
                )

        total_chunks_count = meta.get("total_chunks", len(manager.table) if manager.table is not None else len(results))
        file_count = meta.get("total_files", len(manager.get_indexed_files()))

    else:
        # Fallback to On-The-Fly Chunking and Search
        ext_list = [e.strip() for e in args.ext.split(",")] if args.ext else None
        files = collect_files(target_path, extensions=ext_list)
        if blacklist_filenames:
            files = [
                f for f in files
                if f.name not in blacklist_filenames and str(f) not in blacklist_filenames
            ]
        if not files:
            if args.json:
                sys.stdout.write(format_json_output([], args.query, 0, 0, 0.0, args.mode, args.dim) + "\n")
            else:
                sys.stderr.write(f"No matching text files found in '{args.path}' (after blacklist exclusion).\n")
            return 0

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

        if search_mode == "auto":
            search_mode = infer_mode(files)

        if show_feedback:
            sys.stderr.write(f"[*] Scanned {len(files)} files -> {len(all_chunks)} candidate chunks (on-the-fly)\n")
            if len(files) >= 50:
                sys.stderr.write(f"[Tip] Run 'gemmas index {args.path}' to index this corpus for instant queries.\n")

        # 4. Search Execution (Daemon IPC or in-process)
        if not args.no_daemon:
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
            if args.heat and results and not (args.files_with_matches or args.quiet):
                if show_feedback:
                    sys.stderr.write("[*] Pass 2: Computing line-level saliency heatmaps...\n")
                engine.compute_line_saliency(
                    query=args.query,
                    chunks=[r[0] for r in results],
                    mode=search_mode,
                    dehyphen=args.dehyphen
                )

        if blacklist_filenames and results:
            results = [
                r for r in results
                if Path(r[0]["path"]).name not in blacklist_filenames
                and r[0]["path"] not in blacklist_filenames
            ][:args.limit]

        total_chunks_count = len(all_chunks)
        file_count = len(files)

    elapsed_ms = (time.perf_counter() - start_total) * 1000

    # Output dispatch
    if args.quiet:
        return 0 if (results and len(results) > 0) else 1

    if args.files_with_matches:
        out_str = format_files_only_output(results)
        if out_str:
            sys.stdout.write(out_str + "\n")
        return 0 if results else 1

    if args.compact:
        out_str = format_compact_output(results)
        if out_str:
            sys.stdout.write(out_str + "\n")
        return 0

    if args.interactive:
        if show_feedback and sys.stdout.isatty():
            sys.stderr.write("[*] Launching interactive browser...\n")
        selected_paths = run_interactive_browser(
            results=results,
            query=args.query,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold
        )
        if selected_paths:
            if args.json:
                selected_set = set(selected_paths)
                curated_results = [r for r in results if r[0]["path"] in selected_set]
                sys.stdout.write(format_json_output(
                    results=curated_results,
                    query=args.query,
                    total_chunks=total_chunks_count,
                    file_count=len(selected_paths),
                    elapsed_ms=elapsed_ms,
                    mode=search_mode,
                    dim_size=args.dim,
                    heat_threshold=args.heat_threshold
                ) + "\n")
            elif args.compact:
                selected_set = set(selected_paths)
                curated_results = [r for r in results if r[0]["path"] in selected_set]
                out_str = format_compact_output(curated_results)
                if out_str:
                    sys.stdout.write(out_str + "\n")
            else:
                for p in selected_paths:
                    sys.stdout.write(p + "\n")
            sys.stdout.flush()
            return 0
        return 0 if sys.stdout.isatty() else 1
    elif args.json:
        sys.stdout.write(format_json_output(
            results=results,
            query=args.query,
            total_chunks=total_chunks_count,
            file_count=file_count,
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold
        ) + "\n")
    else:
        sys.stdout.write(format_terminal_output(
            results=results,
            query=args.query,
            total_chunks=total_chunks_count,
            file_count=file_count,
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim,
            heat_threshold=args.heat_threshold,
            plain=args.plain
        ) + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
