# -*- coding: utf-8 -*-
from __future__ import annotations

"""
run_ablation.py
===============

Thin ablation runner for MatSci-RAG.

All modes share the SAME main_pipeline.py backbone. Only the specified component
is disabled/replaced.

Modes
-----
full
    Complete MatSci-RAG.

wo_structure
    Remove structure-aware section localization. Retrieval is performed across
    all section buckets, while association and multimodal evidence remain enabled.

wo_multimodal
    Keep structure-aware retrieval and graph association, but COMPLETELY remove
    figure/table evidence before context construction and disable figure-image
    attachment. Text, equations, and references remain available.

wo_association
    Keep structure-aware retrieval and multimodal evidence, but disable explicit
    text -> figure/table graph association. Figure/table objects can still enter
    independently through caption-level semantic retrieval implemented in
    caption_retriever.py.

Examples
--------
python ablation_component/run_ablation.py \
    --mode full \
    --output_dir examples/test_case/output \
    --query "What are the main factors affecting the gamma-prime phase size?" \
    --task qa

python ablation_component/run_ablation.py \
    --mode wo_structure \
    --output_dir examples/test_case/output \
    --query "What are the main factors affecting the gamma-prime phase size?" \
    --task qa

python ablation_component/run_ablation.py \
    --mode wo_multimodal \
    --output_dir examples/test_case/output \
    --query "What are the main factors affecting the gamma-prime phase size?" \
    --task qa

python ablation_component/run_ablation.py \
    --mode wo_association \
    --output_dir examples/test_case/output \
    --query "What are the main factors affecting the gamma-prime phase size?" \
    --task qa
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow this script to live in repository_root/ablation_component/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_pipeline import (  # noqa: E402
    EvidenceGraph,
    GraphExpander,
    PipelineConfig,
    QueryAnalysis,
    RetrievalHit,
    build_pipeline_from_output_dir,
)
from caption_retriever import IndependentCaptionRetriever  # noqa: E402


ABLATION_MODES = {
    "full",
    "wo_structure",
    "wo_multimodal",
    "wo_association",
}


class AblationExpander:
    """A mode-aware wrapper around the normal GraphExpander.

    It changes only the evidence component required by the selected ablation.
    """

    def __init__(
        self,
        base_expander: GraphExpander,
        mode: str,
        caption_retriever: Optional[IndependentCaptionRetriever] = None,
        caption_top_k: int = 5,
    ) -> None:
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation mode: {mode}")

        self.base_expander = base_expander
        self.mode = mode
        self.caption_retriever = caption_retriever
        self.caption_top_k = max(1, int(caption_top_k))
        self.last_debug: Dict[str, Any] = {}

    @staticmethod
    def _empty_result() -> Dict[str, List[Dict[str, Any]]]:
        return {
            "chunks": [],
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

    @staticmethod
    def _append_caption_hit(
        expanded: Dict[str, List[Dict[str, Any]]],
        hit: Any,
        graph: EvidenceGraph,
    ) -> None:
        node = graph.get_node(hit.node_id)
        if not node:
            return

        bucket = "figures" if hit.node_type == "figure" else "tables"
        if bucket not in expanded:
            return

        uid = graph.make_evidence_uid(hit.node_id)
        existing_uids = {
            str(x.get("evidence_uid", ""))
            for x in expanded.get(bucket, [])
            if isinstance(x, dict)
        }
        if uid in existing_uids:
            return

        node_copy = dict(node)
        node_copy["evidence_uid"] = uid

        # Crucial for the ablation: the object is NOT linked to any retrieved
        # text chunk. It enters only through independent caption matching.
        node_copy["triggered_by_chunk_ids"] = []
        node_copy["ablation_retrieval"] = {
            "mode": "independent_caption_retrieval",
            "caption_similarity": float(hit.score),
            "caption_text": hit.caption_text,
        }
        expanded[bucket].append(node_copy)

    def expand(
        self,
        hits: List[RetrievalHit],
        graph: EvidenceGraph,
        query_analysis: QueryAnalysis,
        enable_association: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:

        if self.mode == "wo_association":
            # Force graph object association OFF.
            expanded = self.base_expander.expand(
                hits=hits,
                graph=graph,
                query_analysis=query_analysis,
                enable_association=False,
            )

            caption_hits = []
            if self.caption_retriever is not None:
                caption_hits = self.caption_retriever.retrieve(
                    query=query_analysis.normalized_query,
                    graph=graph,
                    top_k=self.caption_top_k,
                )
                for hit in caption_hits:
                    self._append_caption_hit(
                        expanded=expanded,
                        hit=hit,
                        graph=graph,
                    )

            self.last_debug = {
                "mode": self.mode,
                "explicit_association_enabled": False,
                "independent_caption_retrieval_enabled": True,
                "caption_top_k": self.caption_top_k,
                "caption_hits": [
                    {
                        "evidence_uid": x.evidence_uid,
                        "node_id": x.node_id,
                        "node_type": x.node_type,
                        "score": x.score,
                    }
                    for x in caption_hits
                ],
            }
            return expanded

        # For full / wo_structure / wo_multimodal, keep the normal graph expansion.
        expanded = self.base_expander.expand(
            hits=hits,
            graph=graph,
            query_analysis=query_analysis,
            enable_association=enable_association,
        )

        if self.mode == "wo_multimodal":
            before = {
                "figures": len(expanded.get("figures", [])),
                "tables": len(expanded.get("tables", [])),
            }

            # Manuscript-aligned ablation: figures, tables, and their associated
            # payloads are completely excluded before evidence budgeting/context.
            expanded["figures"] = []
            expanded["tables"] = []

            self.last_debug = {
                "mode": self.mode,
                "explicit_association_enabled": bool(enable_association),
                "multimodal_objects_removed": True,
                "removed_before_budget": before,
            }
        else:
            self.last_debug = {
                "mode": self.mode,
                "explicit_association_enabled": bool(enable_association),
            }

        return expanded


def mode_flags(mode: str) -> Dict[str, bool]:
    if mode == "full":
        return {
            "enable_structure": True,
            "enable_association": True,
            "enable_multimodal": True,
        }
    if mode == "wo_structure":
        return {
            "enable_structure": False,
            "enable_association": True,
            "enable_multimodal": True,
        }
    if mode == "wo_multimodal":
        return {
            "enable_structure": True,
            "enable_association": True,
            "enable_multimodal": False,
        }
    if mode == "wo_association":
        return {
            "enable_structure": True,
            "enable_association": False,
            "enable_multimodal": True,
        }
    raise ValueError(f"Unsupported ablation mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatSci-RAG full model or one-component ablations."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(ABLATION_MODES),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--task",
        choices=["extraction", "qa"],
        default="qa",
    )

    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-large-en-v1.5",
    )
    parser.add_argument(
        "--rerank_model",
        default="BAAI/bge-reranker-large",
    )
    parser.add_argument(
        "--llm_model",
        default="GLM-4.1V-Thinking-Flash",
    )
    parser.add_argument("--env_file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asset_base_dir", default="")

    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--rerank_top_n", type=int, default=10)

    parser.add_argument("--max_evidence_objects", type=int, default=20)
    parser.add_argument("--max_textual_tokens", type=int, default=5120)
    parser.add_argument("--max_visual_tabular_items", type=int, default=5)

    # Only used by w/o Association.
    parser.add_argument(
        "--caption_top_k",
        type=int,
        default=5,
        help="Independent figure/table caption retrieval depth for w/o Association.",
    )

    parser.add_argument(
        "--allow_generation_fallback",
        action="store_true",
    )
    parser.add_argument("--output", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flags = mode_flags(args.mode)

    config = PipelineConfig(
        output_dir=args.output_dir,
        embedding_model_path=args.embedding_model,
        rerank_model_path=args.rerank_model,
        env_file_path=args.env_file,
        llm_model_name=args.llm_model,
        device=args.device,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_n=args.rerank_top_n,
        max_evidence_objects=args.max_evidence_objects,
        max_textual_evidence_tokens=args.max_textual_tokens,
        max_visual_tabular_items=args.max_visual_tabular_items,
        enable_structure=flags["enable_structure"],
        enable_association=flags["enable_association"],
        enable_multimodal=flags["enable_multimodal"],
        asset_base_dir=args.asset_base_dir,
        allow_generation_fallback=args.allow_generation_fallback,
    )

    pipeline, graph = build_pipeline_from_output_dir(config)

    caption_retriever = None
    if args.mode == "wo_association":
        caption_retriever = IndependentCaptionRetriever(
            embedding_model_path=args.embedding_model,
            device=args.device,
            top_k=args.caption_top_k,
            include_figures=True,
            include_tables=True,
        )

    # Wrap only the evidence-expansion layer. The retriever, reranker, budgets,
    # prompts, generator, assembler, and packager remain shared.
    ablation_expander = AblationExpander(
        base_expander=pipeline.expander,
        mode=args.mode,
        caption_retriever=caption_retriever,
        caption_top_k=args.caption_top_k,
    )
    pipeline.expander = ablation_expander

    result = pipeline.run(
        query=args.query,
        graph=graph,
        task_mode=args.task,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_n=args.rerank_top_n,
    )

    # Explicitly record the ablation definition in every result JSON.
    result["ablation"] = {
        "mode": args.mode,
        "enable_structure": flags["enable_structure"],
        "enable_association": flags["enable_association"],
        "enable_multimodal": flags["enable_multimodal"],
        "caption_level_independent_retrieval": (
            args.mode == "wo_association"
        ),
        "caption_top_k": (
            args.caption_top_k
            if args.mode == "wo_association"
            else None
        ),
        "expander_debug": ablation_expander.last_debug,
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"[OK] {output_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
