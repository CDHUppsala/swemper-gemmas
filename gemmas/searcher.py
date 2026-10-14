import os
# Must set CUDA alloc config before torch initialization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import warnings
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

def dehyphenate(text: str) -> str:
    """
    Soft dehyphenation for historical OCR: joins hyphenated line breaks
    (e.g. 'känne-\\ndom' -> 'kännedom') into single compound words for model encoding.
    """
    return re.sub(r"([^\W\d_]+)[-‐‑\xad]\s*\r?\n\s*([^\W\d_]+)", r"\1\2", text)


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class GemmaSearchEngine:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        device: str = "auto",
        dim: int = 256,
        batch_size: int = 64,
        max_seq_length: int = 512
    ):
        self.device = get_device(device)
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype}
        ).to(self.device)

        # Crucial for preventing CUDA OOMs on large text files:
        # EmbeddingGemma2 config has an unbounded 1e30 default max_seq_length
        self.model.max_seq_length = self.max_seq_length

    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        mode: str = "code",
        limit: int = 5,
        dehyphen: bool = True,
        show_progress: bool = False,
        progress_callback: Any = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not chunks:
            return []

        # 1. Determine prompt names
        q_prompt = "CodeRetrieval" if mode == "code" else "Retrieval"
        doc_prompt = "document"

        # Check model prompts availability
        available = getattr(self.model, "prompts", {})
        q_p = q_prompt if q_prompt in available else None
        d_p = doc_prompt if doc_prompt in available else None

        # 2. Encode query (single forward pass)
        with torch.inference_mode():
            q_vec = self.model.encode(
                [query],
                prompt_name=q_p,
                truncate_dim=self.dim,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # 3. Format & encode candidates
            # Prepend context metadata for code and text
            candidate_texts = []
            apply_dehyphen = dehyphen and mode != "code"
            for c in chunks:
                path_str = c.get("path", "")
                snippet = c.get("text", "")
                if apply_dehyphen:
                    snippet = dehyphenate(snippet)
                if mode == "code":
                    formatted = f"Context: {path_str}:{c.get('start_line', 1)}\nCode:\n{snippet}"
                else:
                    formatted = f"Title: {path_str}\nText:\n{snippet}"
                candidate_texts.append(formatted.strip())

            total_chunks = len(candidate_texts)
            if progress_callback and total_chunks > self.batch_size:
                # Micro-batch encoding with progress callback
                total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
                all_vecs = []
                for b_idx in range(total_batches):
                    start_i = b_idx * self.batch_size
                    end_i = min(start_i + self.batch_size, total_chunks)
                    batch_texts = candidate_texts[start_i:end_i]
                    b_vecs = self.model.encode(
                        batch_texts,
                        prompt_name=d_p,
                        truncate_dim=self.dim,
                        normalize_embeddings=True,
                        batch_size=self.batch_size,
                        show_progress_bar=False
                    )
                    all_vecs.append(b_vecs)
                    progress_callback(b_idx + 1, total_batches, end_i, total_chunks)
                c_vecs = np.vstack(all_vecs)
            else:
                c_vecs = self.model.encode(
                    candidate_texts,
                    prompt_name=d_p,
                    truncate_dim=self.dim,
                    normalize_embeddings=True,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Compute cosine similarities (unit normalized vectors)
        sims = np.dot(c_vecs, q_vec.T).flatten()

        # 5. Rank and return top matches
        ranked_indices = np.argsort(sims)[::-1][:limit]
        
        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            results.append((chunks[idx], round(score, 4)))
            
        return results

    def compute_line_saliency(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        mode: str = "code",
        dehyphen: bool = True
    ) -> None:
        """
        Second pass: Batch-encodes individual lines of top chunks to compute
        fine-grained similarity heatmaps and attaches `chunk['line_scores'] = {offset: score}`.
        """
        if not chunks:
            return

        q_prompt = "CodeRetrieval" if mode == "code" else "Retrieval"
        doc_prompt = "document"

        available = getattr(self.model, "prompts", {})
        q_p = q_prompt if q_prompt in available else None
        d_p = doc_prompt if doc_prompt in available else None

        apply_dehyphen = dehyphen and mode != "code"

        with torch.inference_mode():
            # 1. Encode query
            q_vec = self.model.encode(
                [query],
                prompt_name=q_p,
                truncate_dim=self.dim,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # 2. Gather non-empty lines across all chunks
            all_lines = []
            line_map = []  # (chunk, line_offset, line_str)

            for chunk in chunks:
                chunk["line_scores"] = {}
                text_lines = chunk.get("text", "").splitlines()
                n_lines = len(text_lines)
                for offset, l in enumerate(text_lines):
                    stripped = l.strip()
                    if len(stripped) < 3:
                        continue

                    eval_line = stripped
                    if apply_dehyphen:
                        # 1. Complete trailing hyphen from next line
                        if offset + 1 < n_lines:
                            m_end = re.search(r"([^\W\d_]+)[-‐‑\xad]\s*$", stripped)
                            next_stripped = text_lines[offset + 1].strip()
                            m_next = re.match(r"^([^\W\d_]+)", next_stripped)
                            if m_end and m_next:
                                eval_line = stripped[:m_end.start()] + m_end.group(1) + m_next.group(1)
                        # 2. Complete leading continuation from previous line
                        if offset > 0:
                            prev_stripped = text_lines[offset - 1].strip()
                            m_prev = re.search(r"([^\W\d_]+)[-‐‑\xad]\s*$", prev_stripped)
                            m_start = re.match(r"^([^\W\d_]+)", eval_line)
                            if m_prev and m_start:
                                eval_line = m_prev.group(1) + eval_line

                    all_lines.append(eval_line)
                    line_map.append((chunk, offset, l))

            if not all_lines:
                return

            # 3. Batch encode all lines in a single fast forward pass (<20ms)
            line_vecs = self.model.encode(
                all_lines,
                prompt_name=d_p,
                truncate_dim=self.dim,
                normalize_embeddings=True,
                batch_size=min(64, self.batch_size * 2),
                show_progress_bar=False
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Dot products
        sims = np.dot(line_vecs, q_vec.T).flatten()

        # 5. Populate chunk["line_scores"]
        for (chunk, offset, _), score in zip(line_map, sims):
            chunk["line_scores"][offset] = round(float(score), 4)

