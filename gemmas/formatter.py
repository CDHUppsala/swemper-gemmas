"""
gemmas.formatter - Terminal and JSON presentation for search results.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Simple ANSI colors
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

def c(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m" if USE_COLOR else text

def bold(text: str) -> str:
    return c(text, "1")

def green(text: str) -> str:
    return c(text, "32")

def cyan(text: str) -> str:
    return c(text, "36")

def yellow(text: str) -> str:
    return c(text, "33")

def dim(text: str) -> str:
    return c(text, "2")

def format_score(score: float) -> str:
    s_str = f"{score:.3f}"
    if score >= 0.70:
        return green(s_str)
    elif score >= 0.55:
        return cyan(s_str)
    return yellow(s_str)

def format_files_only_output(results: List[Tuple[Dict[str, Any], float]]) -> str:
    """Print only unique file paths of matching documents, one per line."""
    if not results:
        return ""
    seen = set()
    unique = []
    for chunk, _ in results:
        p = chunk["path"]
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return "\n".join(unique)

def format_compact_output(results: List[Tuple[Dict[str, Any], float]]) -> str:
    """Print each match on a single line (score, path:line, brief snippet)."""
    if not results:
        return ""
    lines = []
    for chunk, score in results:
        rel_path = chunk["path"]
        start = chunk["start_line"]
        text_lines = [l.strip() for l in chunk.get("text", "").splitlines() if l.strip()]
        snippet = text_lines[0] if text_lines else ""
        if len(snippet) > 85:
            snippet = snippet[:82] + "..."
        lines.append(f"{format_score(score)}  {bold(rel_path)}:{start}  {snippet}")
    return "\n".join(lines)

def format_terminal_output(
    results: List[Tuple[Dict[str, Any], float]],
    query: str,
    total_chunks: int,
    file_count: int,
    elapsed_ms: float,
    mode: str,
    dim_size: int,
    heat_threshold: float = 0.70,
    plain: bool = False
) -> str:
    if not results:
        return f"\n{yellow('No matching chunks found for query:')} {bold(repr(query))}\n"

    lines = []
    if not plain:
        lines.append(f"\n{bold('🔍 Gemma Semantic Search Results')} {dim(f'({mode} mode, {dim_size}d)')}")
        lines.append(f"{dim('Query:')} {bold(repr(query))}")
        lines.append("=" * 70)

    for rank, (chunk, score) in enumerate(results, 1):
        rel_path = chunk["path"]
        abs_path = str(Path(rel_path).resolve())
        start = chunk["start_line"]
        end = chunk["end_line"]
        line_scores = chunk.get("line_scores", {})
        
        # Clickable URI
        file_uri = f"file://{abs_path}#L{start}-L{end}"
        header = f"[{rank}] {bold(rel_path)}:{start}–{end}  (Score: {format_score(score)})"
        lines.append(f"\n{header}")
        lines.append(f"    {dim(file_uri)}")
        lines.append("    " + "-" * 62)

        # Print excerpt (up to 14 lines)
        text_lines = chunk["text"].splitlines()
        preview = text_lines[:14]
        has_heat = bool(line_scores)

        for offset, l in enumerate(preview):
            lineno = start + offset
            l_score = line_scores.get(offset) if line_scores.get(offset) is not None else line_scores.get(str(offset), 0.0)

            if has_heat:
                if l_score >= 0.78:
                    # Hot line: Star + Score leftmost, bold gold
                    score_col = c(f"★ {l_score:.3f}", "1;33")
                    text_part = bold(l)
                elif l_score >= heat_threshold:
                    # Warm line: Score leftmost, green
                    score_col = c(f"  {l_score:.3f}", "32")
                    text_part = green(l)
                else:
                    score_col = "       "
                    text_part = l

                prefix = f"    {score_col} {dim('│')} {dim(f'{lineno:>5} │')} "
                lines.append(f"{prefix}{text_part}")
            else:
                lines.append(f"    {dim(f'{lineno:>5} │')} {l}")

        if len(text_lines) > 14:
            remaining = len(text_lines) - 14
            if has_heat:
                lines.append(f"            {dim('│')} {dim('      │')} ... ({remaining} more lines)")
            else:
                lines.append(f"    {dim('      │')} ... ({remaining} more lines)")

    if not plain:
        lines.append("\n" + "=" * 70)
        footer = f"Scanned {bold(str(total_chunks))} chunks across {bold(str(file_count))} files in {bold(f'{elapsed_ms:.1f}ms')}"
        lines.append(footer + "\n")

    return "\n".join(lines)

def format_json_output(
    results: List[Tuple[Dict[str, Any], float]],
    query: str,
    total_chunks: int,
    file_count: int,
    elapsed_ms: float,
    mode: str,
    dim_size: int,
    heat_threshold: float = 0.70
) -> str:
    matches = []
    for rank, (chunk, score) in enumerate(results, 1):
        abs_path = str(Path(chunk["path"]).resolve())
        line_scores = chunk.get("line_scores", {})
        
        highlighted = []
        text_lines = chunk["text"].splitlines()
        for offset, l in enumerate(text_lines):
            l_score = line_scores.get(offset) if line_scores.get(offset) is not None else line_scores.get(str(offset), 0.0)
            if l_score >= heat_threshold:
                highlighted.append({
                    "line_no": chunk["start_line"] + offset,
                    "score": l_score,
                    "text": l
                })

        match_entry = {
            "rank": rank,
            "path": chunk["path"],
            "absolute_path": abs_path,
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "score": score,
            "text": chunk["text"]
        }
        if highlighted:
            match_entry["highlighted_lines"] = highlighted

        matches.append(match_entry)

    payload = {
        "query": query,
        "mode": mode,
        "dim": dim_size,
        "elapsed_ms": round(elapsed_ms, 2),
        "total_chunks_scanned": total_chunks,
        "files_scanned": file_count,
        "match_count": len(matches),
        "matches": matches
    }
    return json.dumps(payload, indent=2)
