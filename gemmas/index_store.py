import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pyarrow as pa
import lancedb

CACHE_INDEX_BASE = Path.home() / ".cache" / "gemmas" / "indexes"

def get_canonical_hash(target_path: Path) -> str:
    resolved = str(target_path.resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]

def get_cache_index_path(target_path: Path) -> Path:
    p = target_path.resolve()
    name = p.name if p.name else "root"
    slug = f"{get_canonical_hash(p)}_{name}"
    return CACHE_INDEX_BASE / f"{slug}.lance"

def get_local_index_path(target_path: Path) -> Path:
    p = target_path.resolve()
    if p.is_dir():
        return p / ".gemmas.lance"
    else:
        return p.parent / f".{p.name}.gemmas.lance"

def resolve_index_path(target_path: Path, explicit_dir: Optional[Path | str] = None) -> Optional[Path]:
    """
    Hierarchical index resolution:
    1. Explicit --index-dir if specified
    2. Local in-tree index (<target_path>/.gemmas.lance)
    3. User central cache (~/.cache/gemmas/indexes/<hash>_<name>.lance)
    Returns Path if a valid index directory exists, else None.
    """
    if explicit_dir is not None:
        p = Path(explicit_dir).resolve()
        if p.exists() and (p / "meta.json").exists():
            return p
        elif p.exists():
            return p
        return None

    local_p = get_local_index_path(target_path)
    if local_p.exists() and (local_p / "meta.json").exists():
        return local_p

    cache_p = get_cache_index_path(target_path)
    if cache_p.exists() and (cache_p / "meta.json").exists():
        return cache_p

    return None

def list_all_indexes() -> List[Dict[str, Any]]:
    """Scan ~/.cache/gemmas/indexes for all known cached indexes."""
    indexes = []
    if CACHE_INDEX_BASE.exists():
        for d in CACHE_INDEX_BASE.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                try:
                    meta = json.loads((d / "meta.json").read_text())
                    meta["index_path"] = str(d)
                    meta["is_cache"] = True
                    total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    meta["disk_size_mb"] = round(total_bytes / (1024 * 1024), 2)
                    indexes.append(meta)
                except Exception:
                    pass
    return sorted(indexes, key=lambda x: x.get("indexed_at", 0), reverse=True)


class LanceIndexManager:
    TABLE_NAME = "chunks"

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path).resolve()
        self.meta_file = self.index_path / "meta.json"
        self._db = None
        self._table = None

    @property
    def db(self):
        if self._db is None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.index_path))
        return self._db

    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self.db.open_table(self.TABLE_NAME)
            except Exception:
                self._table = None
        return self._table

    def get_metadata(self) -> Dict[str, Any]:
        if self.meta_file.exists():
            try:
                return json.loads(self.meta_file.read_text())
            except Exception:
                return {}
        return {}

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        self.meta_file.write_text(json.dumps(meta, indent=2))

    def create_table(self, dim: int = 256) -> None:
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("path", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("file_size", pa.int64()),
        ])
        self._table = self.db.create_table(self.TABLE_NAME, schema=schema, mode="overwrite")

    def add_chunks(self, records: List[Dict[str, Any]], dim: int = 256) -> None:
        if not records:
            return
        if self.table is None:
            self.create_table(dim=dim)
        self.table.add(records)

    def delete_by_paths(self, rel_paths: List[str]) -> None:
        if self.table is None or not rel_paths:
            return
        for p in rel_paths:
            escaped = p.replace("'", "''")
            try:
                self.table.delete(f"path = '{escaped}'")
            except Exception:
                pass

    def get_indexed_files(self) -> Dict[str, Tuple[float, int]]:
        """Returns map of {rel_path: (mtime, file_size)} for all indexed files."""
        if self.table is None or len(self.table) == 0:
            return {}
        try:
            arrow_data = self.table.search().select(["path", "mtime", "file_size"]).to_arrow().to_pydict()
            res = {}
            for p, m, s in zip(arrow_data.get("path", []), arrow_data.get("mtime", []), arrow_data.get("file_size", [])):
                res[p] = (float(m), int(s))
            return res
        except Exception:
            return {}

    def search(
        self,
        query_vec: np.ndarray,
        limit: int = 5,
        where: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.table is None or len(self.table) == 0:
            return []

        q = self.table.search(query_vec).metric("cosine")
        if where:
            q = q.where(where)
        results = q.limit(limit).to_list()

        matches = []
        for r in results:
            dist = float(r.get("_distance", 1.0))
            score = round(1.0 - dist, 4)
            chunk = {
                "id": r.get("id"),
                "path": r.get("path"),
                "start_line": int(r.get("start_line", 1)),
                "end_line": int(r.get("end_line", 1)),
                "text": r.get("text", "")
            }
            matches.append((chunk, score))
        return matches

    def drop(self) -> None:
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
