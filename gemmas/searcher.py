"""
gemmas.searcher - EmbeddingGemma-2 model loading, encoding, and vector scoring.
"""

import os
import warnings
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

# Suppress HF unauthenticated warnings and verbose tokenizer output
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

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
        batch_size: int = 32
    ):
        self.device = get_device(device)
        self.dim = dim
        self.batch_size = batch_size
        
        # Suppress HF unauthenticated warnings if not set
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype}
        ).to(self.device)

    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        mode: str = "code",
        limit: int = 5
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
        for c in chunks:
            path_str = c.get("path", "")
            snippet = c.get("text", "")
            if mode == "code":
                formatted = f"Context: {path_str}:{c.get('start_line', 1)}\nCode:\n{snippet}"
            else:
                formatted = f"Title: {path_str}\nText:\n{snippet}"
            candidate_texts.append(formatted.strip())

        c_vecs = self.model.encode(
            candidate_texts,
            prompt_name=d_p,
            truncate_dim=self.dim,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        # 4. Compute cosine similarities (unit normalized vectors)
        sims = np.dot(c_vecs, q_vec.T).flatten()

        # 5. Rank and return top matches
        ranked_indices = np.argsort(sims)[::-1][:limit]
        
        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            results.append((chunks[idx], round(score, 4)))
            
        return results
