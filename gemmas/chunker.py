"""
gemmas.chunker - File discovery and text/code chunking.
"""

from pathlib import Path
from typing import List, Dict, Any, Set

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

def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in DEFAULT_IGNORED_DIRS:
            return True
        if part.startswith(".") and part not in {".", ".."}:
            # Ignore hidden directories like .cache
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

def chunk_file(file_path: Path, chunk_lines: int = 35, overlap: int = 10) -> List[Dict[str, Any]]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    step = max(1, chunk_lines - overlap)
    chunks = []

    for i in range(0, total_lines, step):
        chunk_slice = lines[i:i + chunk_lines]
        block = "\n".join(chunk_slice).strip()
        # Discard trivially short blocks (fewer than 20 characters)
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

def infer_mode(files: List[Path]) -> str:
    if not files:
        return "code"
    code_count = sum(1 for f in files if f.suffix.lower() in CODE_EXTENSIONS)
    return "code" if code_count >= (len(files) / 2) else "text"
