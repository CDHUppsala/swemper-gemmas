"""
gemmas.interactive - Lightweight, zero-dependency curses TUI for navigating and expanding search results.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import curses
except ImportError:
    curses = None

def run_interactive_browser(
    results: List[Tuple[Dict[str, Any], float]],
    query: str,
    mode: str,
    dim_size: int,
    heat_threshold: float = 0.70
) -> None:
    if not curses:
        sys.stderr.write("Error: 'curses' library is not available on this platform.\n")
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.stderr.write("Interactive mode requires an interactive TTY terminal.\n")
        return

    if not results:
        print(f"\nNo matches found for {query!r} to display in interactive mode.\n")
        return

    def _tui(stdscr):
        try:
            curses.curs_set(0)
        except curses.error:
            pass

        try:
            curses.use_default_colors()
        except curses.error:
            pass

        # Initialize colors
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # High score
            curses.init_pair(2, curses.COLOR_CYAN, -1)    # Medium score
            curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Low score
            curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_CYAN) # Selection bar
            curses.init_pair(5, curses.COLOR_BLUE, -1)    # Line numbers
            curses.init_pair(6, curses.COLOR_MAGENTA, -1) # Headers

        current_idx = 0
        detail_scroll = 0

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            if h < 10 or w < 40:
                stdscr.addstr(0, 0, "Terminal too small for interactive view.")
                stdscr.refresh()
                key = stdscr.getch()
                if key in (ord('q'), 27):
                    break
                continue

            # Header bar
            title = f" GEMMAS SEMANTIC SEARCH — Query: {repr(query)} ({mode} mode, {dim_size}d) "
            try:
                stdscr.addstr(0, 0, title[:w-1], curses.A_REVERSE | curses.A_BOLD)
            except curses.error:
                pass

            # Responsive Layout: Side-by-side if w >= 110, else stacked top/bottom
            is_side_by_side = (w >= 110)

            if is_side_by_side:
                list_w = min(48, w // 3)
                detail_x = list_w + 1
                detail_w = w - detail_x - 1
                list_h = h - 3
                detail_h = h - 3

                # Vertical separator
                for y in range(1, h - 2):
                    try:
                        stdscr.addch(y, list_w, curses.ACS_VLINE)
                    except curses.error:
                        pass
            else:
                list_w = w - 1
                list_h = max(4, min(len(results) + 2, (h - 4) // 2))
                detail_x = 0
                detail_w = w - 1
                detail_h = h - list_h - 4

                # Horizontal separator
                sep_y = list_h + 1
                try:
                    stdscr.addstr(sep_y, 0, "─" * (w - 1), curses.color_pair(5) if curses.has_colors() else curses.A_DIM)
                except curses.error:
                    pass

            # 1. Render Result List
            for i, (chunk, score) in enumerate(results):
                row_y = 1 + i
                if is_side_by_side and row_y >= list_h + 1:
                    break
                if not is_side_by_side and row_y >= list_h + 1:
                    break

                is_selected = (i == current_idx)
                fname = Path(chunk["path"]).name
                score_str = f"{score:.3f}"
                line_range = f"{chunk['start_line']}–{chunk['end_line']}"
                line_text = f"[{i+1}] {score_str}  {fname} ({line_range})"

                # Truncate if exceeds list width
                if len(line_text) > list_w - 2:
                    line_text = line_text[:list_w - 5] + "..."

                if is_selected:
                    attr = curses.color_pair(4) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE
                    line_text = f" > {line_text:<{list_w - 4}}"
                else:
                    attr = curses.A_NORMAL
                    line_text = f"   {line_text:<{list_w - 4}}"

                try:
                    stdscr.addstr(row_y, 0, line_text[:list_w - 1], attr)
                except curses.error:
                    pass

            # 2. Render Selected Chunk Detail Pane
            sel_chunk, sel_score = results[current_idx]
            d_y = 1 if is_side_by_side else (list_h + 2)

            # Metadata header
            score_col = curses.color_pair(1) if sel_score >= 0.70 else (curses.color_pair(2) if sel_score >= 0.55 else curses.color_pair(3))
            header_str = f"Match #{current_idx + 1} | Score: {sel_score:.4f} | {sel_chunk['path']}:{sel_chunk['start_line']}–{sel_chunk['end_line']}"
            try:
                stdscr.addstr(d_y, detail_x, header_str[:detail_w - 1], curses.A_BOLD)
                d_y += 1
                abs_path = str(Path(sel_chunk["path"]).resolve())
                uri_str = f"URI: file://{abs_path}#L{sel_chunk['start_line']}-L{sel_chunk['end_line']}"
                stdscr.addstr(d_y, detail_x, uri_str[:detail_w - 1], curses.color_pair(5) if curses.has_colors() else curses.A_DIM)
                d_y += 1
            except curses.error:
                pass

            # Chunk Lines with scroll support
            chunk_lines = sel_chunk["text"].splitlines()
            line_scores = sel_chunk.get("line_scores", {})
            visible_code_lines = detail_h - 2

            # Clamp detail scroll
            max_scroll = max(0, len(chunk_lines) - visible_code_lines)
            detail_scroll = max(0, min(detail_scroll, max_scroll))

            slice_start = detail_scroll
            slice_end = min(len(chunk_lines), slice_start + visible_code_lines)

            has_heat = bool(line_scores)

            for offset, l in enumerate(chunk_lines[slice_start:slice_end]):
                curr_y = d_y + offset
                if curr_y >= h - 2:
                    break
                line_idx = slice_start + offset
                line_no = sel_chunk["start_line"] + line_idx
                l_score = line_scores.get(line_idx) if line_scores.get(line_idx) is not None else line_scores.get(str(line_idx), 0.0)

                if has_heat:
                    if l_score >= 0.78:
                        score_prefix = f"★ {l_score:.3f} │ "
                        score_attr = (curses.color_pair(3) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                        code_attr = (curses.color_pair(1) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                    elif l_score >= heat_threshold:
                        score_prefix = f"  {l_score:.3f} │ "
                        score_attr = curses.color_pair(2) if curses.has_colors() else curses.A_DIM
                        code_attr = (curses.color_pair(2) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                    else:
                        score_prefix = "        │ "
                        score_attr = curses.color_pair(5) if curses.has_colors() else curses.A_DIM
                        code_attr = curses.A_NORMAL

                    lineno_str = f"{line_no:>5} │ "
                    lineno_attr = curses.color_pair(5) if curses.has_colors() else curses.A_DIM
                    code_str = l.replace("\t", "    ")

                    try:
                        stdscr.addstr(curr_y, detail_x, score_prefix, score_attr)
                        stdscr.addstr(curr_y, detail_x + len(score_prefix), lineno_str, lineno_attr)
                        avail_w = detail_w - len(score_prefix) - len(lineno_str) - 1
                        if avail_w > 0:
                            stdscr.addstr(curr_y, detail_x + len(score_prefix) + len(lineno_str), code_str[:avail_w], code_attr)
                    except curses.error:
                        pass
                else:
                    lineno_str = f"{line_no:>5} │ "
                    lineno_attr = curses.color_pair(5) if curses.has_colors() else curses.A_DIM
                    code_str = l.replace("\t", "    ")
                    try:
                        stdscr.addstr(curr_y, detail_x, lineno_str, lineno_attr)
                        avail_w = detail_w - len(lineno_str) - 1
                        if avail_w > 0:
                            stdscr.addstr(curr_y, detail_x + len(lineno_str), code_str[:avail_w], curses.A_NORMAL)
                    except curses.error:
                        pass

            if len(chunk_lines) > visible_code_lines:
                scroll_indicator = f" [Lines {slice_start + 1}–{slice_end} of {len(chunk_lines)} (Use PgUp/PgDn/w/s to scroll)] "
                try:
                    stdscr.addstr(h - 3, detail_x, scroll_indicator[:detail_w - 1], curses.A_DIM)
                except curses.error:
                    pass

            # 3. Footer instructions
            footer = " ↑/↓ (j/k): Select Match  │  w/s (PgUp/PgDn): Scroll Detail  │  o/e: Open in $EDITOR  │  q/ESC: Quit "
            try:
                stdscr.addstr(h - 1, 0, footer[:w - 1], curses.A_REVERSE)
            except curses.error:
                pass

            stdscr.refresh()

            # Handle user input
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break

            if ch in (ord('q'), ord('Q'), 27):  # q or ESC
                break
            elif ch in (curses.KEY_UP, ord('k'), ord('K')):
                if current_idx > 0:
                    current_idx -= 1
                    detail_scroll = 0
            elif ch in (curses.KEY_DOWN, ord('j'), ord('J')):
                if current_idx < len(results) - 1:
                    current_idx += 1
                    detail_scroll = 0
            elif ch in (curses.KEY_NPAGE, ord('s'), ord('S'), ord(' ')): # Page down / scroll down
                detail_scroll += 5
            elif ch in (curses.KEY_PPAGE, ord('w'), ord('W')): # Page up / scroll up
                detail_scroll = max(0, detail_scroll - 5)
            elif ch in (ord('o'), ord('e'), 10, 13): # Enter or o/e to open editor
                editor = os.environ.get("EDITOR") or "nano"
                start_l = sel_chunk["start_line"]
                target_file = sel_chunk["path"]

                curses.endwin()
                try:
                    # Try opening editor at line
                    if "vim" in editor or "nano" in editor or "vi" in editor:
                        subprocess.run([editor, f"+{start_l}", target_file])
                    else:
                        subprocess.run([editor, target_file])
                except Exception as ex:
                    print(f"Error opening editor '{editor}': {ex}")
                    input("Press Enter to return...")
                
                # Re-initialize curses screen upon return
                stdscr.clear()
                curses.doupdate()

    try:
        curses.wrapper(_tui)
    except Exception as e:
        sys.stderr.write(f"\nWarning: Interactive curses mode could not run ({e}). Falling back to standard output.\n")
        from .formatter import format_terminal_output
        print(format_terminal_output(
            results=results,
            query=query,
            total_chunks=len(results),
            file_count=len(set(r[0]['path'] for r in results)),
            elapsed_ms=0.0,
            mode=mode,
            dim_size=dim_size
        ))
