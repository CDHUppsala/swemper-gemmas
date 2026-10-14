"""
gemmas.cli - Command-line interface for Gemma semantic search.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

from . import __version__
from .chunker import collect_files, chunk_file, infer_mode
from .searcher import GemmaSearchEngine
from .formatter import format_terminal_output, format_json_output

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemmas",
        description="Fast on-the-fly semantic code and text search powered by Google EmbeddingGemma-2."
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
        "--chunk-lines",
        type=int,
        default=35,
        help="Sliding chunk window size in lines (default: 35)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Overlap between consecutive chunks in lines (default: 10)"
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
        all_chunks.extend(chunk_file(f, chunk_lines=args.chunk_lines, overlap=args.overlap))

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

    # 4. Initialize search engine and compute scores
    engine = GemmaSearchEngine(
        model_name=args.model,
        device=args.device,
        dim=args.dim
    )

    results = engine.search(
        query=args.query,
        chunks=all_chunks,
        mode=search_mode,
        limit=args.limit
    )

    elapsed_ms = (time.perf_counter() - start_total) * 1000

    # 5. Output
    if args.json:
        sys.stdout.write(format_json_output(
            results=results,
            query=args.query,
            total_chunks=len(all_chunks),
            file_count=len(files),
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim
        ) + "\n")
    else:
        sys.stdout.write(format_terminal_output(
            results=results,
            query=args.query,
            total_chunks=len(all_chunks),
            file_count=len(files),
            elapsed_ms=elapsed_ms,
            mode=search_mode,
            dim_size=args.dim
        ) + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
