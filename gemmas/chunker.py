"""
gemmas.chunker - File discovery and text/code chunking.
Supports both semantic sentence-aware paragraph chunking and classic sliding line windowing.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".gemini",
    ".cargo",
    "target",
}

CODE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".rs", ".go", ".c", ".cpp", ".h", ".hpp",
    ".java", ".rb", ".php", ".sh", ".bash",
    ".lua", ".sql", ".html", ".css", ".json",
    ".yaml", ".yml", ".toml"
}

# Swedish, Latin, and common abbreviations to protect during sentence splitting
ABBREV_RE = re.compile(
    r'\b(?:t\.\s*ex|o\.\s*s\.\s*v|bl\.\s*a|d\.\s*v\.\s*s|t\.\s*o\.\s*m|f\.\s*o\.\s*m|'
    r'o\.\s*d|m\.\s*fl|d\.\s*y|d\.\s*ä|s\.\s*k|fr\.\s*o\.\s*m|t\.\s*v|'
    r'dr|prof|hr|fru|frkn|nr|fig|tab|sid|kap|vol|ca|etc|i\.e|e\.g)\.',
    re.IGNORECASE
)
INITIAL_RE = re.compile(r'\b[A-ZÅÄÖ]\.')

def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in DEFAULT_IGNORED_DIRS:
            return True
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False

def is_text_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return False
        return True
    except Exception:
        return False

def collect_files(target: Path, extensions: List[str] | None = None) -> List[Path]:
    if target.is_file():
        return [target] if is_text_file(target) else []

    if not target.is_dir():
        return []

    ext_set = {f".{e.lstrip('.')}" for e in extensions} if extensions else None
    collected = []

    for p in target.rglob("*"):
        if not p.is_file():
            continue
        if is_ignored(p):
            continue
        if ext_set and p.suffix.lower() not in ext_set:
            continue
        if is_text_file(p):
            collected.append(p)

    return sorted(collected)

def collect_blacklist_filenames(items: List[Path | str]) -> Set[str]:
    """
    Collects blacklisted filenames from directories, list files (.txt/.csv), or direct filenames.
    - If item is an existing directory: recursively collects all filenames in the directory tree.
    - If item is an existing file: adds the file's basename and parses lines (for .txt/.csv list files).
    - If item does not exist locally: treats item as a direct filename or path to exclude.
    """
    blacklist: Set[str] = set()
    for item in items:
        p = Path(item).resolve()
        if p.exists():
            if p.is_dir():
                for root, dirs_list, files in os.walk(p):
                    dirs_list[:] = [
                        sub for sub in dirs_list
                        if sub not in DEFAULT_IGNORED_DIRS and not sub.startswith(".")
                    ]
                    for f in files:
                        if not f.startswith("."):
                            blacklist.add(f)
            elif p.is_file():
                blacklist.add(p.name)
                blacklist.add(str(p))
                blacklist.add(str(item))
                # Attempt to read lines in case this is a blacklist text or CSV file
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = [line] if "," not in line else [part.strip().strip('"\'') for part in line.split(",")]
                            for part in parts:
                                if part and len(part) < 500:
                                    blacklist.add(Path(part).name)
                                    blacklist.add(part)
                except Exception:
                    pass
        else:
            raw_str = str(item).strip().strip('"\'')
            if raw_str:
                blacklist.add(Path(raw_str).name)
                blacklist.add(raw_str)
    return blacklist

def infer_mode(files: List[Path]) -> str:
    if not files:
        return "code"
    code_count = sum(1 for f in files if f.suffix.lower() in CODE_EXTENSIONS)
    return "code" if code_count >= (len(files) / 2) else "text"

# --- Sentence-Aware Paragraph Chunking ---

def _extract_sentences_from_paragraph(
    para_lines: List[Tuple[int, str]]
) -> List[Tuple[int, int, str]]:
    """
    Extracts individual sentences from a paragraph while tracking exact 1-indexed
    start_line and end_line in the original source document.
    """
    if not para_lines:
        return []

    char_to_line = []
    text_chars = []
    for l_no, l_str in para_lines:
        for ch in l_str:
            char_to_line.append(l_no)
            text_chars.append(ch)
        char_to_line.append(l_no)
        text_chars.append("\n")

    full_text = "".join(text_chars)

    # Mask periods in abbreviations and initials so they aren't split on
    masked = full_text
    for m in ABBREV_RE.finditer(full_text):
        masked = masked[:m.start()] + masked[m.start():m.end()].replace(".", "@") + masked[m.end():]
    for m in INITIAL_RE.finditer(masked):
        masked = masked[:m.start()] + masked[m.start():m.end()].replace(".", "@") + masked[m.end():]

    # Sentence boundary: terminal punctuation followed by whitespace and capital letter or end of paragraph
    pattern = re.compile(r'([.!?]+)(?:\s+(?=[A-ZÅÄÖ0-9"\'\-])|\s*$)')
    sentences = []
    last_idx = 0

    for match in pattern.finditer(masked):
        end_idx = match.end()
        s_text = full_text[last_idx:end_idx].strip()
        if s_text:
            s_start_line = char_to_line[last_idx]
            s_end_line = char_to_line[min(end_idx - 1, len(char_to_line) - 1)]
            sentences.append((s_start_line, s_end_line, s_text))
        last_idx = end_idx

    if last_idx < len(full_text):
        rem = full_text[last_idx:].strip()
        if rem:
            sentences.append((char_to_line[last_idx], char_to_line[-1], rem))

    return sentences

def chunk_by_sentences(
    lines: List[str],
    file_path: Path,
    min_sentences: int = 5,
    sentence_overlap: int = 3,
    max_sentences: int = 15,
    max_lines: int = 55
) -> List[Dict[str, Any]]:
    """
    Chunks text by accumulating sentences across paragraph (newline) boundaries
    until >= min_sentences is reached, overlapping by sentence_overlap.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return []

    # 1. Group lines into paragraphs (delimited by blank lines)
    paragraphs: List[List[Tuple[int, str]]] = []
    curr_para: List[Tuple[int, str]] = []

    for idx, l in enumerate(lines, 1):
        if l.strip():
            curr_para.append((idx, l))
        else:
            if curr_para:
                paragraphs.append(curr_para)
                curr_para = []
    if curr_para:
        paragraphs.append(curr_para)

    if not paragraphs:
        return []

    # 2. Extract sentences per paragraph and assign paragraph IDs
    all_sentences: List[Tuple[int, int, str]] = []
    sentence_para_id: List[int] = []

    for p_id, p in enumerate(paragraphs):
        p_sentences = _extract_sentences_from_paragraph(p)
        for s in p_sentences:
            all_sentences.append(s)
            sentence_para_id.append(p_id)

    num_sentences = len(all_sentences)
    if num_sentences == 0:
        return []

    # If the entire file has fewer sentences than min_sentences, return 1 chunk
    if num_sentences <= min_sentences:
        block = "\n".join(lines).strip()
        if len(block) >= 20:
            return [{
                "path": str(file_path),
                "start_line": 1,
                "end_line": total_lines,
                "line_count": total_lines,
                "sentence_count": num_sentences,
                "text": block
            }]
        return []

    chunks = []
    curr_s_idx = 0

    while curr_s_idx < num_sentences:
        # Check termination: if remaining sentences are fewer than min_sentences
        # and previous chunk already covered to the end, stop
        if chunks and (num_sentences - curr_s_idx) < min_sentences and chunks[-1]["end_line"] == total_lines:
            break

        chunk_s_indices = [curr_s_idx]
        next_s_idx = curr_s_idx + 1

        while next_s_idx < num_sentences:
            chunk_s_indices.append(next_s_idx)

            cur_start_l = all_sentences[chunk_s_indices[0]][0]
            cur_end_l = all_sentences[chunk_s_indices[-1]][1]
            span_lines = cur_end_l - cur_start_l + 1

            # Check boundary condition:
            # 1. We have at least min_sentences
            # 2. We are at a paragraph boundary (next sentence belongs to a new paragraph or is last)
            is_para_boundary = (next_s_idx == num_sentences - 1) or (
                sentence_para_id[next_s_idx] != sentence_para_id[next_s_idx + 1]
            )
            has_min = len(chunk_s_indices) >= min_sentences

            # Safety cap to avoid exceeding EmbeddingGemma's context limit
            hit_safety_cap = len(chunk_s_indices) >= max_sentences or span_lines >= max_lines

            if (has_min and is_para_boundary) or hit_safety_cap:
                break

            next_s_idx += 1

        start_line = all_sentences[chunk_s_indices[0]][0]
        end_line = all_sentences[chunk_s_indices[-1]][1]

        # Extract text slice directly from original lines preserving formatting
        chunk_slice = lines[start_line - 1 : end_line]
        block = "\n".join(chunk_slice).strip()

        if len(block) >= 20:
            chunks.append({
                "path": str(file_path),
                "start_line": start_line,
                "end_line": end_line,
                "line_count": len(chunk_slice),
                "sentence_count": len(chunk_s_indices),
                "text": block
            })

        if chunk_s_indices[-1] >= num_sentences - 1 or end_line >= total_lines:
            break

        # Slide window by sentence overlap
        advance = max(1, len(chunk_s_indices) - sentence_overlap)
        curr_s_idx += advance

    return chunks

# --- Line-Based Sliding Window Chunking ---

def chunk_by_lines(
    lines: List[str],
    file_path: Path,
    chunk_lines: int = 35,
    overlap: int = 3
) -> List[Dict[str, Any]]:
    """
    Classic fixed-line sliding window chunker.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return []

    step = max(1, chunk_lines - overlap)
    chunks = []

    for i in range(0, total_lines, step):
        chunk_slice = lines[i:i + chunk_lines]
        block = "\n".join(chunk_slice).strip()
        if len(block) >= 20:
            chunks.append({
                "path": str(file_path),
                "start_line": i + 1,
                "end_line": min(i + len(chunk_slice), total_lines),
                "line_count": len(chunk_slice),
                "text": block
            })

        if i + chunk_lines >= total_lines:
            break

    return chunks

# --- Universal Dispatcher ---

def chunk_file(
    file_path: Path,
    strategy: str = "sentences",
    chunk_lines: int = 35,
    overlap: int = 3,
    min_sentences: int = 5,
    sentence_overlap: int = 3,
    max_sentences: int = 15,
    max_lines: int = 55
) -> List[Dict[str, Any]]:
    """
    Chunks a file using the specified strategy:
    - 'sentences': Paragraph newline-aligned sentence accumulation (default).
    - 'lines': Classic fixed-line sliding window.
    - 'auto': Uses 'lines' for source code, 'sentences' for prose/documents.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    if not lines:
        return []

    eff_strategy = strategy
    if eff_strategy == "auto":
        eff_strategy = "lines" if file_path.suffix.lower() in CODE_EXTENSIONS else "sentences"

    if eff_strategy == "sentences":
        return chunk_by_sentences(
            lines=lines,
            file_path=file_path,
            min_sentences=min_sentences,
            sentence_overlap=sentence_overlap,
            max_sentences=max_sentences,
            max_lines=max_lines
        )
    else:
        return chunk_by_lines(
            lines=lines,
            file_path=file_path,
            chunk_lines=chunk_lines,
            overlap=overlap
        )
