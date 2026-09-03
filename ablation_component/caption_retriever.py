# -*- coding: utf-8 -*-
from __future__ import annotations

"""
caption_retriever.py
====================

Independent caption-level figure/table retriever used ONLY for the
"w/o Association" ablation.

Purpose
-------
The full MatSci-RAG pipeline obtains figures/tables through explicit
text -> figure/table graph associations. In the "w/o Association" ablation,
those graph associations are disabled, but figure/table evidence is still
allowed to enter through independent caption-level semantic matching.

This module therefore:
1. collects figure/table nodes from paper_graph.json;
2. embeds ONLY their labels/captions (not graph-neighbor information);
3. independently retrieves the most semantically similar figure/table objects.

It intentionally does not use:
- cites_figure / cites_table edges;
- nearby_evidence_ids;
- triggered_by_chunk_ids;
- section-to-object graph association.

The actual figure image / table payload is still assembled later by
main_pipeline.RenderAssembler.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()


@dataclass
class CaptionHit:
    evidence_uid: str
    node_id: str
    node_type: str
    score: float
    caption_text: str


class IndependentCaptionRetriever:
    """Brute-force caption retriever for figure/table nodes.

    The corpus is small at the single-paper level, so an extra FAISS index is not
    necessary. Embeddings are cached after the first query for a graph.
    """

    def __init__(
        self,
        embedding_model_path: str,
        device: str = "cuda",
        top_k: int = 5,
        include_figures: bool = True,
        include_tables: bool = True,
    ) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embedding_model_path = embedding_model_path
        self.device = device
        self.top_k = max(1, int(top_k))
        self.include_figures = bool(include_figures)
        self.include_tables = bool(include_tables)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={
                "device": device,
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        self._cache_key: Optional[Tuple[str, str, int]] = None
        self._items: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None

    @staticmethod
    def _graph_cache_key(graph: Any) -> Tuple[str, str, int]:
        return (
            _safe_str(getattr(graph, "doc_id", "")),
            _safe_str(getattr(graph, "source_file", "")),
            len(getattr(graph, "nodes", []) or []),
        )

    @staticmethod
    def _caption_text(node: Dict[str, Any]) -> str:
        """Build caption-level text without using graph association metadata."""
        node_type = _safe_str(node.get("node_type"))
        md = _safe_dict(node.get("metadata"))

        title = _safe_str(node.get("title"))
        content = _safe_str(node.get("content"))

        if node_type == "table":
            caption = _safe_str(md.get("caption")) or content
        else:
            caption = content

        parts = [x for x in (title, caption) if _normalize_text(x)]
        return _normalize_text(" ".join(parts))

    def _build_cache(self, graph: Any) -> None:
        key = self._graph_cache_key(graph)
        if self._cache_key == key and self._matrix is not None:
            return

        items: List[Dict[str, Any]] = []
        texts: List[str] = []

        for node in getattr(graph, "nodes", []) or []:
            if not isinstance(node, dict):
                continue

            node_type = _safe_str(node.get("node_type"))
            if node_type == "figure" and not self.include_figures:
                continue
            if node_type == "table" and not self.include_tables:
                continue
            if node_type not in {"figure", "table"}:
                continue

            node_id = _safe_str(node.get("node_id")).strip()
            if not node_id:
                continue

            caption_text = self._caption_text(node)
            if not caption_text:
                continue

            evidence_uid = graph.make_evidence_uid(node_id)

            items.append({
                "evidence_uid": evidence_uid,
                "node_id": node_id,
                "node_type": node_type,
                "caption_text": caption_text,
            })
            texts.append(caption_text)

        self._items = items
        self._cache_key = key

        if not texts:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            return

        vectors = np.asarray(
            self.embeddings.embed_documents(texts),
            dtype=np.float32,
        )

        # Defensive normalization even though the embedding adapter is configured
        # with normalize_embeddings=True.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = vectors / norms

    def retrieve(
        self,
        query: str,
        graph: Any,
        top_k: Optional[int] = None,
    ) -> List[CaptionHit]:
        query = _normalize_text(query)
        if not query:
            return []

        self._build_cache(graph)

        if self._matrix is None or self._matrix.size == 0 or not self._items:
            return []

        query_vector = np.asarray(
            self.embeddings.embed_query(query),
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(query_vector))
        if norm <= 0:
            return []
        query_vector = query_vector / norm

        scores = self._matrix @ query_vector
        k = min(max(1, int(top_k or self.top_k)), len(self._items))
        ranked_indices = np.argsort(-scores)[:k]

        hits: List[CaptionHit] = []
        for idx in ranked_indices:
            item = self._items[int(idx)]
            hits.append(
                CaptionHit(
                    evidence_uid=item["evidence_uid"],
                    node_id=item["node_id"],
                    node_type=item["node_type"],
                    score=float(scores[int(idx)]),
                    caption_text=item["caption_text"],
                )
            )
        return hits
