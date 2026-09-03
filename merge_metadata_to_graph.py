# -*- coding: utf-8 -*-
from __future__ import annotations

"""
merge_metadata_to_graph.py
==========================

Merge the ``metadata_structured`` node produced by metadata_processor.py into an
existing ``paper_graph.json`` while enforcing document/interface consistency.

Key guarantees
--------------
- ``node_id`` remains document-local: ``metadata_structured``.
- ``evidence_uid`` is the interface-level identifier:
  ``<doc_id>::metadata_structured``.
- Cross-document metadata merges are rejected by default.
- Existing structured metadata is replaced deterministically rather than duplicated.
- Graph statistics and enrichment status are refreshed after the merge.
- Basic node_id/evidence_uid uniqueness is checked before writing the graph.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==========================================================
# 0. IO
# ==========================================================

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ==========================================================
# 1. Basic helpers
# ==========================================================

def safe_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def safe_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else []


def safe_str(obj: Any) -> str:
    return obj if isinstance(obj, str) else ""


def make_evidence_uid(doc_id: str, node_id: str) -> str:
    doc_id = safe_str(doc_id).strip()
    node_id = safe_str(node_id).strip()
    if not doc_id or not node_id:
        return ""
    return f"{doc_id}::{node_id}"


def ensure_graph_shell(graph: Dict[str, Any]) -> Dict[str, Any]:
    if "nodes" not in graph or not isinstance(graph["nodes"], list):
        graph["nodes"] = []
    if "edges" not in graph or not isinstance(graph["edges"], list):
        graph["edges"] = []
    if "stats" not in graph or not isinstance(graph["stats"], dict):
        graph["stats"] = {}
    return graph


# ==========================================================
# 2. Metadata-node normalization and validation
# ==========================================================

def normalize_metadata_node_for_graph(
    metadata_node: Dict[str, Any],
    graph: Dict[str, Any],
    allow_source_file_mismatch: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a normalized copy of the metadata node and non-fatal warnings."""
    node = copy.deepcopy(safe_dict(metadata_node))
    warnings: List[str] = []

    graph_doc_id = safe_str(graph.get("doc_id")).strip()
    if not graph_doc_id:
        raise ValueError("paper_graph.json is missing a non-empty doc_id.")

    if safe_str(node.get("node_type")).strip() != "metadata_structured":
        raise ValueError(
            "metadata_node.json node_type must be 'metadata_structured'; "
            f"got {node.get('node_type')!r}."
        )

    # The public schema uses a fixed local node_id for structured document metadata.
    node_id = safe_str(node.get("node_id")).strip() or "metadata_structured"
    if node_id != "metadata_structured":
        raise ValueError(
            "metadata_node.json node_id must be 'metadata_structured'; "
            f"got {node_id!r}."
        )
    node["node_id"] = node_id

    node_doc_id = safe_str(node.get("doc_id")).strip()
    if node_doc_id and node_doc_id != graph_doc_id:
        raise ValueError(
            "Refusing cross-document metadata merge: "
            f"graph doc_id={graph_doc_id!r}, metadata doc_id={node_doc_id!r}."
        )
    node["doc_id"] = graph_doc_id

    expected_uid = make_evidence_uid(graph_doc_id, node_id)
    existing_uid = safe_str(node.get("evidence_uid")).strip()
    if existing_uid and existing_uid != expected_uid:
        raise ValueError(
            "metadata evidence_uid is inconsistent with graph doc_id/node_id: "
            f"expected {expected_uid!r}, got {existing_uid!r}."
        )
    node["evidence_uid"] = expected_uid

    graph_source = safe_str(graph.get("source_file")).strip()
    node_source = safe_str(node.get("source_file")).strip()
    if graph_source and node_source and graph_source != node_source:
        message = (
            "source_file differs between graph and metadata node: "
            f"graph={graph_source!r}, metadata={node_source!r}."
        )
        if not allow_source_file_mismatch:
            raise ValueError(message + " Use --allow-source-file-mismatch only if this is intentional.")
        warnings.append(message)
    if graph_source:
        node["source_file"] = graph_source

    metadata = safe_dict(node.get("metadata"))
    metadata["evidence_uid"] = expected_uid
    node["metadata"] = metadata

    required = ["content", "metadata"]
    missing = [key for key in required if key not in node]
    if missing:
        raise ValueError(f"metadata_node.json is missing required fields: {missing}")

    return node, warnings


def find_existing_metadata_node_indices(nodes: List[Dict[str, Any]]) -> List[int]:
    indices: List[int] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if (
            node.get("node_id") == "metadata_structured"
            or node.get("node_type") == "metadata_structured"
        ):
            indices.append(i)
    return indices


# ==========================================================
# 3. Graph consistency/stat refresh
# ==========================================================

def validate_identifier_uniqueness(graph: Dict[str, Any]) -> Dict[str, Any]:
    node_ids: Dict[str, int] = {}
    evidence_uids: Dict[str, int] = {}

    for node in safe_list(graph.get("nodes")):
        if not isinstance(node, dict):
            continue
        node_id = safe_str(node.get("node_id")).strip()
        evidence_uid = safe_str(node.get("evidence_uid")).strip()
        if node_id:
            node_ids[node_id] = node_ids.get(node_id, 0) + 1
        if evidence_uid:
            evidence_uids[evidence_uid] = evidence_uids.get(evidence_uid, 0) + 1

    duplicate_node_ids = sorted([key for key, count in node_ids.items() if count > 1])
    duplicate_evidence_uids = sorted([key for key, count in evidence_uids.items() if count > 1])

    if duplicate_node_ids or duplicate_evidence_uids:
        raise ValueError(
            "Identifier uniqueness check failed. "
            f"duplicate node_id={duplicate_node_ids[:10]}, "
            f"duplicate evidence_uid={duplicate_evidence_uids[:10]}"
        )

    return {
        "node_id_unique": True,
        "evidence_uid_unique": True,
        "node_id_count": len(node_ids),
        "evidence_uid_count": len(evidence_uids),
    }


def refresh_graph_stats(graph: Dict[str, Any]) -> None:
    nodes = safe_list(graph.get("nodes"))
    edges = safe_list(graph.get("edges"))
    stats = safe_dict(graph.get("stats"))

    # Preserve chunking-related statistics that json_split.py already calculated,
    # but refresh counts affected by enrichment.
    stats["metadata_nodes"] = sum(
        1 for n in nodes if isinstance(n, dict) and n.get("node_type") == "metadata"
    )
    stats["metadata_structured_nodes"] = sum(
        1 for n in nodes if isinstance(n, dict) and n.get("node_type") == "metadata_structured"
    )
    stats["total_nodes"] = len(nodes)
    stats["total_edges"] = len(edges)
    graph["stats"] = stats


# ==========================================================
# 4. Merge logic
# ==========================================================

def merge_metadata_node_into_graph(
    graph_data: Dict[str, Any],
    metadata_node: Dict[str, Any],
    replace_existing: bool = True,
    update_enrichment_status: bool = True,
    allow_source_file_mismatch: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    graph = ensure_graph_shell(copy.deepcopy(safe_dict(graph_data)))
    normalized_node, warnings = normalize_metadata_node_for_graph(
        metadata_node=metadata_node,
        graph=graph,
        allow_source_file_mismatch=allow_source_file_mismatch,
    )

    nodes = safe_list(graph.get("nodes"))
    existing_indices = find_existing_metadata_node_indices(nodes)

    report: Dict[str, Any] = {
        "graph_doc_id": graph.get("doc_id"),
        "graph_source_file": graph.get("source_file"),
        "metadata_doc_id": normalized_node.get("doc_id"),
        "metadata_source_file": normalized_node.get("source_file"),
        "metadata_evidence_uid": normalized_node.get("evidence_uid"),
        "existing_metadata_structured_nodes": len(existing_indices),
        "action": "",
        "replaced_indices": [],
        "inserted": False,
        "warnings": warnings,
    }

    if existing_indices and not replace_existing:
        report["action"] = "skip_existing"
    else:
        if existing_indices:
            existing_set = set(existing_indices)
            nodes = [node for idx, node in enumerate(nodes) if idx not in existing_set]
            report["action"] = "replace_existing"
            report["replaced_indices"] = existing_indices
        else:
            report["action"] = "insert_new"

        # Deterministic placement makes graph inspection simpler while preserving
        # all original json_split nodes.
        nodes.insert(0, normalized_node)
        graph["nodes"] = nodes
        report["inserted"] = True

    if update_enrichment_status:
        status = safe_dict(graph.get("graph_enrichment_status"))
        status["metadata_structured"] = True
        status["metadata_evidence_uid"] = normalized_node.get("evidence_uid")
        graph["graph_enrichment_status"] = status

    refresh_graph_stats(graph)
    report["identifier_check"] = validate_identifier_uniqueness(graph)
    report["final_total_nodes"] = len(safe_list(graph.get("nodes")))
    report["final_total_edges"] = len(safe_list(graph.get("edges")))
    report["metadata_structured_nodes"] = safe_dict(graph.get("stats")).get(
        "metadata_structured_nodes", 0
    )

    return graph, report


# ==========================================================
# 5. Output path
# ==========================================================

def build_output_path(
    graph_path: str | Path,
    output_path: Optional[str | Path],
    inplace: bool,
) -> Path:
    graph_path = Path(graph_path)
    if inplace:
        return graph_path
    if output_path is not None:
        return Path(output_path)
    stem = graph_path.stem
    suffix = graph_path.suffix or ".json"
    return graph_path.with_name(f"{stem}_with_metadata{suffix}")


# ==========================================================
# 6. CLI
# ==========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge metadata_structured evidence into a MatSci-RAG paper graph."
    )
    parser.add_argument("--graph", required=True, help="Input paper_graph.json path.")
    parser.add_argument("--metadata-node", required=True, help="metadata_node.json path.")
    parser.add_argument("--output", default=None, help="Output graph path if not using --inplace.")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Update the input paper_graph.json in place (recommended before embedding_json.py).",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not replace an existing metadata_structured node.",
    )
    parser.add_argument(
        "--no-status-update",
        action="store_true",
        help="Do not update graph_enrichment_status.",
    )
    parser.add_argument(
        "--allow-source-file-mismatch",
        action="store_true",
        help="Allow graph/source metadata provenance labels to differ (doc_id must still match).",
    )
    parser.add_argument("--report", default=None, help="Optional path for merge_report.json.")
    args = parser.parse_args()

    graph_data = read_json(args.graph)
    metadata_node = read_json(args.metadata_node)

    merged_graph, report = merge_metadata_node_into_graph(
        graph_data=graph_data,
        metadata_node=metadata_node,
        replace_existing=not args.no_replace,
        update_enrichment_status=not args.no_status_update,
        allow_source_file_mismatch=args.allow_source_file_mismatch,
    )

    output_path = build_output_path(
        graph_path=args.graph,
        output_path=args.output,
        inplace=args.inplace,
    )
    write_json(output_path, merged_graph)

    if args.report:
        write_json(args.report, report)

    print("[OK] metadata_structured merged into graph")
    print(f"[OUT] {output_path}")
    print(f"[ACTION] {report['action']}")
    print(f"[EVIDENCE_UID] {report['metadata_evidence_uid']}")
    print(f"[TOTAL_NODES] {report['final_total_nodes']}")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
