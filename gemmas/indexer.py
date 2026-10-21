import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np

from .chunker import collect_files, chunk_file, infer_mode
from .searcher import GemmaSearchEngine, dehyphenate
from .index_store import (
    LanceIndexManager,
    get_cache_index_path,
    get_local_index_path,
    resolve_index_path
)
from .daemon import (
    ensure_daemon,
    daemon_encode_chunks_rpc
)

def ensure_gitignore(target_dir: Path) -> None:
    """Ensure .gemmas.lance is ignored in local directory's .gitignore."""
    gitignore_path = target_dir / ".gitignore"
    entry = ".gemmas.lance"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        if entry not in content:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Gemmas local vector index\n{entry}\n")
    else:
        gitignore_path.write_text(f"# Gemmas local vector index\n{entry}\n", encoding="utf-8")

def build_index(
    target_path: Path,
    index_dir: Optional[Path | str] = None,
    is_local: bool = False,
    update: bool = False,
    overwrite: bool = False,
    extensions: Optional[List[str]] = None,
    strategy: str = "sentences",
    min_sentences: int = 5,
    sentence_overlap: int = 3,
    chunk_lines: int = 35,
    overlap: int = 3,
    mode: str = "auto",
    dim: int = 256,
    dehyphen: bool = True,
    batch_size: int = 64,
    max_seq_len: int = 512,
    model_name: str = "google/embeddinggemma-2",
    device: str = "auto",
    cooldown: int = 300,
    no_daemon: bool = False,
    show_feedback: bool = True,
    blacklist_filenames: Optional[Set[str]] = None
) -> Optional[Path]:
    target_resolved = target_path.resolve()
    if not target_resolved.exists():
        sys.stderr.write(f"Error: Target path '{target_path}' does not exist.\n")
        return None

    # 1. Resolve destination path
    if index_dir is not None:
        dest_path = Path(index_dir).resolve()
    elif is_local:
        dest_path = get_local_index_path(target_resolved)
    else:
        dest_path = get_cache_index_path(target_resolved)

    # 2. Add gitignore entry if local
    if is_local and target_resolved.is_dir():
        try:
            ensure_gitignore(target_resolved)
        except OSError:
            pass

    manager = LanceIndexManager(dest_path)

    # 3. Handle overwrite
    if overwrite and dest_path.exists():
        if show_feedback:
            sys.stderr.write(f"[*] Overwrite requested: removing existing index at '{dest_path}'\n")
        manager.drop()
        manager = LanceIndexManager(dest_path)

    # 4. Discover files
    files = collect_files(target_resolved, extensions=extensions)
    if blacklist_filenames:
        files = [
            f for f in files
            if f.name not in blacklist_filenames and str(f) not in blacklist_filenames
        ]
    if not files:
        sys.stderr.write(f"[-] No matching text files found in '{target_path}'.\n")
        return None

    search_mode = mode
    if search_mode == "auto":
        search_mode = infer_mode(files)

    # 5. Check indexed files for incremental changes
    indexed_map = manager.get_indexed_files()
    files_to_chunk: List[Path] = []
    files_to_delete: List[str] = []
    rel_to_abs: Dict[str, Path] = {}

    for f in files:
        try:
            rel_str = str(f.relative_to(target_resolved))
        except ValueError:
            rel_str = f.name
        rel_to_abs[rel_str] = f
        try:
            st = f.stat()
        except OSError:
            continue

        if rel_str in indexed_map:
            cached_mtime, cached_size = indexed_map[rel_str]
            if abs(cached_mtime - st.st_mtime) < 0.01 and cached_size == st.st_size:
                continue  # Unchanged file
            else:
                files_to_delete.append(rel_str)
                files_to_chunk.append(f)
        else:
            files_to_chunk.append(f)

    # Check for files deleted from filesystem
    if indexed_map:
        current_rels = set(rel_to_abs.keys())
        deleted = [r for r in indexed_map.keys() if r not in current_rels]
        if deleted:
            files_to_delete.extend(deleted)

    # Apply deletions
    if files_to_delete:
        if show_feedback:
            sys.stderr.write(f"[*] Purging {len(files_to_delete)} stale/modified files from index...\n")
        manager.delete_by_paths(files_to_delete)

    if not files_to_chunk and not files_to_delete and indexed_map:
        total_chunks = len(manager.table) if manager.table is not None else 0
        if show_feedback:
            sys.stderr.write(
                f"[*] Index at '{dest_path}' is up-to-date ({len(indexed_map)} files, {total_chunks} chunks).\n"
            )
        return dest_path

    # 6. Chunk new and modified files
    if show_feedback:
        action_word = "Updating" if indexed_map else "Indexing"
        sys.stderr.write(
            f"[*] {action_word} {len(files_to_chunk)} files (out of {len(files)} total) with strategy '{strategy}'...\n"
        )

    all_chunks = []
    for f in files_to_chunk:
        chunks = chunk_file(
            f,
            strategy=strategy,
            chunk_lines=chunk_lines,
            overlap=overlap,
            min_sentences=min_sentences,
            sentence_overlap=sentence_overlap
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        if show_feedback:
            sys.stderr.write("[-] No non-empty chunks found to index.\n")
        return dest_path

    # 7. Format candidate strings for embedding
    apply_dehyphen = dehyphen and search_mode != "code"
    candidate_texts = []
    for c in all_chunks:
        path_str = c.get("path", "")
        snippet = c.get("text", "")
        if apply_dehyphen:
            snippet = dehyphenate(snippet)
        if search_mode == "code":
            formatted = f"Context: {path_str}:{c.get('start_line', 1)}\nCode:\n{snippet}"
        else:
            formatted = f"Title: {path_str}\nText:\n{snippet}"
        candidate_texts.append(formatted.strip())

    total_chunks = len(candidate_texts)
    if show_feedback:
        sys.stderr.write(f"[*] Encoding {total_chunks} chunks using Gemma (dim: {dim}, batch_size: {batch_size})...\n")

    # 8. Encode chunks via Daemon or Local Fallback
    vecs = None
    if not no_daemon:
        ready = ensure_daemon(
            cooldown=cooldown,
            model_name=model_name,
            device=device,
            dim=dim,
            batch_size=batch_size,
            max_seq_length=max_seq_len,
            verbose=show_feedback
        )
        if ready:
            vecs = daemon_encode_chunks_rpc(
                texts=candidate_texts,
                dim=dim,
                batch_size=batch_size,
                show_feedback=show_feedback,
                timeout=max(300.0, total_chunks * 0.5)
            )

    if vecs is None:
        if show_feedback and not no_daemon:
            sys.stderr.write("[-] Daemon encoding unavailable, falling back to local engine.\n")
        engine = GemmaSearchEngine(
            model_name=model_name,
            device=device,
            dim=dim,
            batch_size=batch_size,
            max_seq_length=max_seq_len
        )
        def on_local_progress(b: int, total_b: int, done: int, total: int) -> None:
            if show_feedback:
                pct = int(done / total * 100) if total else 100
                bar_w = 25
                filled = int(bar_w * (done / total)) if total else bar_w
                bar = "█" * filled + "░" * (bar_w - filled)
                sys.stderr.write(f"\r\033[K[*] Local encoding: [{bar}] {pct}% ({done}/{total} chunks)")
                sys.stderr.flush()

        vecs = engine.encode_candidate_texts(
            candidate_texts=candidate_texts,
            batch_size=batch_size,
            progress_callback=on_local_progress
        )
        if show_feedback:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    # 9. Store into LanceDB
    records = []
    for c, vec in zip(all_chunks, vecs):
        full_path = Path(c["path"])
        try:
            rel_p = str(full_path.relative_to(target_resolved))
        except ValueError:
            rel_p = full_path.name
        try:
            st = full_path.stat()
            mtime = float(st.st_mtime)
            fsize = int(st.st_size)
        except OSError:
            mtime = 0.0
            fsize = 0

        v_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
        records.append({
            "id": f"{rel_p}:{c.get('start_line', 1)}",
            "vector": v_list,
            "path": rel_p,
            "start_line": int(c.get("start_line", 1)),
            "end_line": int(c.get("end_line", 1)),
            "text": c.get("text", ""),
            "mtime": mtime,
            "file_size": fsize,
        })

    manager.add_chunks(records, dim=dim)

    # 10. Save metadata
    total_in_index = len(manager.table) if manager.table is not None else len(records)
    total_files_in_index = len(manager.get_indexed_files())
    meta = {
        "source_root": str(target_resolved),
        "model_name": model_name,
        "dim": dim,
        "strategy": strategy,
        "min_sentences": min_sentences,
        "sentence_overlap": sentence_overlap,
        "chunk_lines": chunk_lines,
        "overlap": overlap,
        "mode": search_mode,
        "dehyphen": dehyphen,
        "indexed_at": time.time(),
        "total_chunks": total_in_index,
        "total_files": total_files_in_index,
        "index_path": str(dest_path)
    }
    manager.save_metadata(meta)

    # Calculate index size
    total_bytes = sum(f.stat().st_size for f in dest_path.rglob("*") if f.is_file())
    mb = round(total_bytes / (1024 * 1024), 2)

    if show_feedback:
        sys.stderr.write(
            f"[*] Successfully indexed {total_chunks} new/updated chunks into '{dest_path}' "
            f"(Total: {total_in_index} chunks, {total_files_in_index} files, {mb} MB).\n"
        )

    return dest_path
