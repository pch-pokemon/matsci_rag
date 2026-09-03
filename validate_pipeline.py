# -*- coding: utf-8 -*-
from __future__ import annotations

"""
validate_pipeline.py
====================
Consistency validator for a built MatSci-RAG single-document database.

The validator is intentionally dependency-light. Core checks use only the Python
standard library; optional deeper checks use ``faiss`` and ``transformers`` when
explicitly requested and available.

Primary targets
---------------
1. paper_graph.json
   - node_id / evidence_uid uniqueness and consistency
   - edge endpoint integrity
   - section/chunk structural references
   - text -> figure/table/equation/reference association consistency
   - figure/table/equation payload sanity
   - stored graph statistics
2. retrieval_docs.jsonl
   - one-to-one coverage of section_chunk nodes
   - doc/source/evidence identifiers
   - text and stored token-count consistency
3. faiss_section_indexes/
   - manifest provenance and configuration consistency
   - expected per-bucket document counts
   - stale/missing .faiss/.pkl pairs
   - optional binary index ntotal/dimension checks via faiss
4. Optional generated result JSON
   - claim support_evidence_uids resolve to graph nodes
   - render_ids resolve and match their requested node types
   - render_payload evidence_uids resolve
5. Local assets
   - figure image paths are resolvable when local images are declared

Exit codes
----------
0 : validation passed (warnings may remain)
1 : one or more validation errors
2 : fatal input/read error

Example
-------
python validate_pipeline.py --output_dir examples/test_case/output

With Doc2X assets explicitly anchored:

python validate_pipeline.py \
    --output_dir examples/test_case/output \
    --asset_base_dir examples/test_case/output

Optional deeper checks:

python validate_pipeline.py \
    --output_dir examples/test_case/output \
    --deep_faiss \
    --deep_token_check \
    --tokenizer_model BAAI/bge-large-en-v1.5
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
# Constants shared with embedding_json.py / main_pipeline.py semantics
# -----------------------------------------------------------------------------

SECTION_BUCKETS: Tuple[str, ...] = (
    "intro",
    "method",
    "results",
    "discussion",
    "results_discussion",
    "conclusion",
    "other",
)

INDEX_NAMES: Dict[str, str] = {
    "intro": "faiss_intro",
    "method": "faiss_method",
    "results": "faiss_results",
    "discussion": "faiss_discussion",
    "results_discussion": "faiss_results_discussion",
    "conclusion": "faiss_conclusion",
    "other": "faiss_other",
    "metadata": "faiss_metadata",
}

KNOWN_NODE_TYPES: Set[str] = {
    "metadata",
    "metadata_structured",
    "section",
    "section_chunk",
    "figure",
    "table",
    "equation",
    "reference",
    "additional_info",
}

EXPECTED_STATS_BY_TYPE: Dict[str, str] = {
    "metadata": "metadata_nodes",
    "metadata_structured": "metadata_structured_nodes",
    "section": "section_nodes",
    "section_chunk": "chunk_nodes",
    "figure": "figure_nodes",
    "table": "table_nodes",
    "equation": "equation_nodes",
    "reference": "reference_nodes",
    "additional_info": "additional_nodes",
}

CITE_RELATION_TO_TYPE: Dict[str, str] = {
    "cites_figure": "figure",
    "cites_table": "table",
    "cites_equation": "equation",
    "cites_reference": "reference",
}

RENDER_KEY_TO_TYPE: Dict[str, str] = {
    "figures": "figure",
    "tables": "table",
    "equations": "equation",
    "references": "reference",
}


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------


def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {p}")
    return obj


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}, line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be an object: {p}, line {line_no}")
            rows.append(obj)
    return rows


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Report model
# -----------------------------------------------------------------------------


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    schema_version: str = "1.0"
    passed: bool = True
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    checks: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **context: Any,
    ) -> None:
        severity = severity.lower().strip()
        if severity not in {"error", "warning", "info"}:
            raise ValueError(f"Unsupported severity: {severity}")
        self.findings.append(Finding(severity, code, message, context))
        if severity == "error":
            self.errors += 1
            self.passed = False
        elif severity == "warning":
            self.warnings += 1
        else:
            self.infos += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "summary": {
                "errors": self.errors,
                "warnings": self.warnings,
                "infos": self.infos,
            },
            "checks": self.checks,
            "findings": [asdict(x) for x in self.findings],
        }


# -----------------------------------------------------------------------------
# Shared section bucketing logic
# -----------------------------------------------------------------------------


def normalize_section_bucket(section_title: str, section_path: str = "") -> str:
    title = (section_title or "").strip().lower()
    path = (section_path or "").strip().lower()
    full = " ".join(f"{path} {title}".split())

    combined = (
        "results and discussion",
        "results & discussion",
        "discussion and results",
        "discussion & results",
        "results discussion",
    )
    if any(x in full for x in combined):
        return "results_discussion"

    if any(x in full for x in ("introduction", "background")):
        return "intro"

    method_keywords = (
        "method",
        "methods",
        "methodology",
        "materials and methods",
        "materials & methods",
        "experimental",
        "experimental procedure",
        "experimental procedures",
        "experimental details",
        "experimental methods",
        "experiments",
        "materials",
        "procedure",
        "procedures",
    )
    if any(x in full for x in method_keywords):
        return "method"

    if any(x in full for x in ("results", "findings")):
        return "results"

    if any(x in full for x in ("discussion", "general discussion")):
        return "discussion"

    conclusion_keywords = (
        "conclusion",
        "conclusions",
        "concluding remarks",
        "summary",
        "summary and outlook",
        "summary & outlook",
        "outlook",
        "perspectives",
        "final remarks",
    )
    if any(x in full for x in conclusion_keywords):
        return "conclusion"

    return "other"


# -----------------------------------------------------------------------------
# Generic utility checks
# -----------------------------------------------------------------------------


def duplicates(values: Iterable[str]) -> List[str]:
    counter = Counter(x for x in values if x)
    return sorted([x for x, count in counter.items() if count > 1])


def expected_uid(doc_id: str, node_id: str) -> str:
    return f"{doc_id}::{node_id}"


def is_remote_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https", "data"}


def normalize_text_for_compare(text: Any) -> str:
    s = safe_str(text).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split())


def _node_type_index(nodes: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        out[safe_str(node.get("node_type"))].append(node)
    return dict(out)


# -----------------------------------------------------------------------------
# Graph checks
# -----------------------------------------------------------------------------


def validate_graph(
    graph: Dict[str, Any],
    report: ValidationReport,
    graph_path: Path,
    asset_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    doc_id = safe_str(graph.get("doc_id")).strip()
    source_file = safe_str(graph.get("source_file")).strip()
    nodes = [x for x in safe_list(graph.get("nodes")) if isinstance(x, dict)]
    edges = [x for x in safe_list(graph.get("edges")) if isinstance(x, dict)]
    stats = safe_dict(graph.get("stats"))
    chunking = safe_dict(graph.get("chunking_config"))

    if not doc_id:
        report.add("error", "GRAPH_DOC_ID_MISSING", "paper_graph.json has no doc_id")
    if not source_file:
        report.add("warning", "GRAPH_SOURCE_FILE_MISSING", "paper_graph.json has no source_file")
    if not nodes:
        report.add("error", "GRAPH_NODES_EMPTY", "paper_graph.json contains no nodes")

    node_ids = [safe_str(n.get("node_id")).strip() for n in nodes]
    evidence_uids = [safe_str(n.get("evidence_uid")).strip() for n in nodes]
    edge_ids = [safe_str(e.get("edge_id")).strip() for e in edges]

    missing_node_id = [i for i, x in enumerate(node_ids) if not x]
    if missing_node_id:
        report.add(
            "error",
            "NODE_ID_MISSING",
            "One or more graph nodes have no node_id",
            indices=missing_node_id[:20],
            count=len(missing_node_id),
        )

    dup_node_ids = duplicates(node_ids)
    if dup_node_ids:
        report.add(
            "error",
            "NODE_ID_DUPLICATE",
            "Duplicate document-local node_id values detected",
            values=dup_node_ids[:50],
            count=len(dup_node_ids),
        )

    missing_uids = [node_ids[i] for i, x in enumerate(evidence_uids) if not x and i < len(node_ids)]
    if missing_uids:
        report.add(
            "error",
            "EVIDENCE_UID_MISSING",
            "Nodes are missing interface-level evidence_uid values",
            node_ids=missing_uids[:50],
            count=len(missing_uids),
        )

    dup_uids = duplicates(evidence_uids)
    if dup_uids:
        report.add(
            "error",
            "EVIDENCE_UID_DUPLICATE",
            "Duplicate evidence_uid values detected",
            values=dup_uids[:50],
            count=len(dup_uids),
        )

    if doc_id:
        bad_uid_pairs: List[Dict[str, str]] = []
        for node, node_id, uid in zip(nodes, node_ids, evidence_uids):
            if not node_id or not uid:
                continue
            exp = expected_uid(doc_id, node_id)
            if uid != exp:
                bad_uid_pairs.append({"node_id": node_id, "evidence_uid": uid, "expected": exp})
        if bad_uid_pairs:
            report.add(
                "error",
                "EVIDENCE_UID_INCONSISTENT",
                "evidence_uid must equal doc_id::node_id for every node",
                examples=bad_uid_pairs[:30],
                count=len(bad_uid_pairs),
            )

    unknown_types = sorted(
        {
            safe_str(n.get("node_type"))
            for n in nodes
            if safe_str(n.get("node_type")) and safe_str(n.get("node_type")) not in KNOWN_NODE_TYPES
        }
    )
    if unknown_types:
        report.add(
            "warning",
            "NODE_TYPE_UNKNOWN",
            "Graph contains node types not recognized by this validator",
            node_types=unknown_types,
        )

    node_by_id = {safe_str(n.get("node_id")): n for n in nodes if safe_str(n.get("node_id"))}
    uid_to_node = {safe_str(n.get("evidence_uid")): n for n in nodes if safe_str(n.get("evidence_uid"))}
    node_types = _node_type_index(nodes)

    # Node-level provenance consistency.
    provenance_mismatches: List[Dict[str, str]] = []
    for node in nodes:
        nid = safe_str(node.get("node_id"))
        ndoc = safe_str(node.get("doc_id")).strip()
        nsource = safe_str(node.get("source_file")).strip()
        if doc_id and ndoc and ndoc != doc_id:
            provenance_mismatches.append({"node_id": nid, "field": "doc_id", "value": ndoc, "expected": doc_id})
        if source_file and nsource and nsource != source_file:
            provenance_mismatches.append({"node_id": nid, "field": "source_file", "value": nsource, "expected": source_file})
    if provenance_mismatches:
        report.add(
            "error",
            "NODE_PROVENANCE_MISMATCH",
            "Node provenance does not match graph-level provenance",
            examples=provenance_mismatches[:30],
            count=len(provenance_mismatches),
        )

    # Edge identity and endpoint integrity.
    missing_edge_ids = [i for i, x in enumerate(edge_ids) if not x]
    if missing_edge_ids:
        report.add(
            "error",
            "EDGE_ID_MISSING",
            "One or more graph edges have no edge_id",
            indices=missing_edge_ids[:20],
            count=len(missing_edge_ids),
        )
    dup_edge_ids = duplicates(edge_ids)
    if dup_edge_ids:
        report.add(
            "error",
            "EDGE_ID_DUPLICATE",
            "Duplicate edge_id values detected",
            values=dup_edge_ids[:50],
            count=len(dup_edge_ids),
        )

    dangling: List[Dict[str, str]] = []
    missing_relations: List[str] = []
    relation_type_errors: List[Dict[str, str]] = []
    for edge in edges:
        eid = safe_str(edge.get("edge_id"))
        src = safe_str(edge.get("source"))
        tgt = safe_str(edge.get("target"))
        rel = safe_str(edge.get("relation"))
        if not rel:
            missing_relations.append(eid)
        if src not in node_by_id or tgt not in node_by_id:
            dangling.append({"edge_id": eid, "source": src, "target": tgt, "relation": rel})
            continue
        expected_target_type = CITE_RELATION_TO_TYPE.get(rel)
        if expected_target_type and safe_str(node_by_id[tgt].get("node_type")) != expected_target_type:
            relation_type_errors.append(
                {
                    "edge_id": eid,
                    "relation": rel,
                    "target": tgt,
                    "target_type": safe_str(node_by_id[tgt].get("node_type")),
                    "expected_target_type": expected_target_type,
                }
            )
        if rel in CITE_RELATION_TO_TYPE and safe_str(node_by_id[src].get("node_type")) != "section_chunk":
            relation_type_errors.append(
                {
                    "edge_id": eid,
                    "relation": rel,
                    "source": src,
                    "source_type": safe_str(node_by_id[src].get("node_type")),
                    "expected_source_type": "section_chunk",
                }
            )

    if missing_relations:
        report.add(
            "error",
            "EDGE_RELATION_MISSING",
            "Edges without a relation were detected",
            edge_ids=missing_relations[:50],
            count=len(missing_relations),
        )
    if dangling:
        report.add(
            "error",
            "EDGE_DANGLING_ENDPOINT",
            "Edges reference source/target node_ids that do not exist",
            examples=dangling[:30],
            count=len(dangling),
        )
    if relation_type_errors:
        report.add(
            "error",
            "EDGE_RELATION_TYPE_MISMATCH",
            "One or more typed citation edges connect incompatible node types",
            examples=relation_type_errors[:30],
            count=len(relation_type_errors),
        )

    # Section / parent references.
    bad_section_refs: List[Dict[str, str]] = []
    for node in nodes:
        nid = safe_str(node.get("node_id"))
        ntype = safe_str(node.get("node_type"))
        sid = safe_str(node.get("section_id")).strip()
        pid = safe_str(node.get("parent_section_id")).strip()
        if sid:
            sec = node_by_id.get(sid)
            if sec is None or safe_str(sec.get("node_type")) != "section":
                bad_section_refs.append({"node_id": nid, "field": "section_id", "value": sid})
        if pid:
            parent = node_by_id.get(pid)
            if parent is None or safe_str(parent.get("node_type")) != "section":
                bad_section_refs.append({"node_id": nid, "field": "parent_section_id", "value": pid})
        if ntype == "section_chunk" and not sid:
            bad_section_refs.append({"node_id": nid, "field": "section_id", "value": ""})
    if bad_section_refs:
        report.add(
            "error",
            "SECTION_REFERENCE_INVALID",
            "section_id / parent_section_id references are inconsistent",
            examples=bad_section_refs[:30],
            count=len(bad_section_refs),
        )

    # Chunk sequence references and token declarations.
    chunks = node_types.get("section_chunk", [])
    max_declared = chunking.get("max_chunk_tokens")
    token_errors: List[Dict[str, Any]] = []
    bad_chunk_links: List[Dict[str, str]] = []
    for chunk in chunks:
        nid = safe_str(chunk.get("node_id"))
        md = safe_dict(chunk.get("metadata"))
        token_count = md.get("token_count")
        if isinstance(token_count, (int, float)) and isinstance(max_declared, (int, float)):
            if token_count > max_declared:
                token_errors.append({"node_id": nid, "token_count": token_count, "max_chunk_tokens": max_declared})
        for field_name in ("prev_chunk_id", "next_chunk_id"):
            ref = safe_str(md.get(field_name)).strip()
            if ref:
                ref_node = node_by_id.get(ref)
                if ref_node is None or safe_str(ref_node.get("node_type")) != "section_chunk":
                    bad_chunk_links.append({"node_id": nid, "field": field_name, "value": ref})
        for ref in safe_list(md.get("nearby_evidence_ids")):
            ref_s = safe_str(ref)
            if ref_s and ref_s not in node_by_id:
                bad_chunk_links.append({"node_id": nid, "field": "nearby_evidence_ids", "value": ref_s})

    if token_errors:
        report.add(
            "error",
            "CHUNK_TOKEN_LIMIT_EXCEEDED",
            "Stored chunk token_count exceeds graph chunking_config.max_chunk_tokens",
            examples=token_errors[:30],
            count=len(token_errors),
        )
    if bad_chunk_links:
        report.add(
            "error",
            "CHUNK_LINK_INVALID",
            "Chunk metadata contains unresolved prev/next/nearby node references",
            examples=bad_chunk_links[:30],
            count=len(bad_chunk_links),
        )

    # Check prev/next symmetry where both are stored.
    asymmetry: List[Dict[str, str]] = []
    for chunk in chunks:
        nid = safe_str(chunk.get("node_id"))
        md = safe_dict(chunk.get("metadata"))
        nxt = safe_str(md.get("next_chunk_id")).strip()
        if nxt and nxt in node_by_id:
            other_md = safe_dict(node_by_id[nxt].get("metadata"))
            if safe_str(other_md.get("prev_chunk_id")).strip() != nid:
                asymmetry.append({"node_id": nid, "next_chunk_id": nxt})
        prev = safe_str(md.get("prev_chunk_id")).strip()
        if prev and prev in node_by_id:
            other_md = safe_dict(node_by_id[prev].get("metadata"))
            if safe_str(other_md.get("next_chunk_id")).strip() != nid:
                asymmetry.append({"node_id": nid, "prev_chunk_id": prev})
    if asymmetry:
        report.add(
            "warning",
            "CHUNK_LINK_ASYMMETRIC",
            "prev_chunk_id / next_chunk_id metadata are not fully symmetric",
            examples=asymmetry[:30],
            count=len(asymmetry),
        )

    # Validate declared evidence references against available objects and cite edges.
    fig_label_to_id = {
        safe_str(n.get("title")).strip(): safe_str(n.get("node_id"))
        for n in node_types.get("figure", [])
        if safe_str(n.get("title")).strip()
    }
    table_label_to_id = {
        safe_str(n.get("title")).strip(): safe_str(n.get("node_id"))
        for n in node_types.get("table", [])
        if safe_str(n.get("title")).strip()
    }
    edge_triplets = {
        (safe_str(e.get("source")), safe_str(e.get("relation")), safe_str(e.get("target")))
        for e in edges
    }
    unresolved_mentions: List[Dict[str, str]] = []
    missing_cite_edges: List[Dict[str, str]] = []

    for chunk in chunks:
        nid = safe_str(chunk.get("node_id"))
        md = safe_dict(chunk.get("metadata"))
        for label in safe_list(md.get("mentioned_figures")):
            label_s = safe_str(label).strip()
            target = fig_label_to_id.get(label_s)
            if not target:
                unresolved_mentions.append({"node_id": nid, "kind": "figure", "mention": label_s})
            elif (nid, "cites_figure", target) not in edge_triplets:
                missing_cite_edges.append({"node_id": nid, "kind": "figure", "target": target})
        for label in safe_list(md.get("mentioned_tables")):
            label_s = safe_str(label).strip()
            target = table_label_to_id.get(label_s)
            if not target:
                unresolved_mentions.append({"node_id": nid, "kind": "table", "mention": label_s})
            elif (nid, "cites_table", target) not in edge_triplets:
                missing_cite_edges.append({"node_id": nid, "kind": "table", "target": target})
        for eq_id in safe_list(md.get("mentioned_equation_ids")):
            eq_s = safe_str(eq_id).strip()
            if eq_s and (eq_s not in node_by_id or safe_str(node_by_id[eq_s].get("node_type")) != "equation"):
                unresolved_mentions.append({"node_id": nid, "kind": "equation", "mention": eq_s})
            elif eq_s and (nid, "cites_equation", eq_s) not in edge_triplets:
                missing_cite_edges.append({"node_id": nid, "kind": "equation", "target": eq_s})
        for ref_id in safe_list(md.get("mentioned_references")):
            ref_s = safe_str(ref_id).strip()
            if ref_s and (ref_s not in node_by_id or safe_str(node_by_id[ref_s].get("node_type")) != "reference"):
                unresolved_mentions.append({"node_id": nid, "kind": "reference", "mention": ref_s})
            elif ref_s and (nid, "cites_reference", ref_s) not in edge_triplets:
                missing_cite_edges.append({"node_id": nid, "kind": "reference", "target": ref_s})

    if unresolved_mentions:
        # Scientific papers often mention supplemental/unsupported objects that were
        # not parsed into the current document. Treat as warning, not graph corruption.
        report.add(
            "warning",
            "MENTION_UNRESOLVED",
            "Some explicit figure/table/equation/reference mentions could not be resolved to graph nodes",
            examples=unresolved_mentions[:40],
            count=len(unresolved_mentions),
        )
    if missing_cite_edges:
        report.add(
            "error",
            "MENTION_EDGE_MISSING",
            "A resolvable evidence mention lacks its expected chunk citation edge",
            examples=missing_cite_edges[:40],
            count=len(missing_cite_edges),
        )

    # Figure/table payload sanity + local image resolution.
    asset_roots: List[Path] = []
    if asset_base_dir is not None:
        asset_roots.append(asset_base_dir.resolve())
    asset_roots.extend(
        [
            graph_path.parent.resolve(),
            graph_path.parent.parent.resolve(),
            (graph_path.parent.parent / "intermediate").resolve(),
        ]
    )
    # Deduplicate roots while retaining order.
    seen_roots: Set[str] = set()
    asset_roots = [r for r in asset_roots if not (str(r) in seen_roots or seen_roots.add(str(r)))]

    missing_images: List[Dict[str, Any]] = []
    no_image_locator: List[str] = []
    for fig in node_types.get("figure", []):
        nid = safe_str(fig.get("node_id"))
        md = safe_dict(fig.get("metadata"))
        image_path = safe_str(md.get("image_path")).strip()
        image_url = safe_str(md.get("image_url")).strip()
        locator = image_path or image_url
        if not locator:
            no_image_locator.append(nid)
            continue
        if is_remote_uri(locator):
            continue
        p = Path(locator)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [(root / p).resolve() for root in asset_roots]
        if not any(c.exists() and c.is_file() for c in candidates):
            missing_images.append(
                {
                    "node_id": nid,
                    "declared": locator,
                    "tried": [str(c) for c in candidates[:6]],
                }
            )
    if no_image_locator:
        report.add(
            "warning",
            "FIGURE_IMAGE_LOCATOR_MISSING",
            "Figure nodes without image_path/image_url were found; captions remain usable but image-level multimodal access is unavailable",
            node_ids=no_image_locator[:50],
            count=len(no_image_locator),
        )
    if missing_images:
        report.add(
            "error",
            "FIGURE_IMAGE_NOT_FOUND",
            "Local figure image paths declared in the graph cannot be resolved",
            examples=missing_images[:30],
            count=len(missing_images),
        )

    malformed_tables: List[Dict[str, Any]] = []
    for tab in node_types.get("table", []):
        nid = safe_str(tab.get("node_id"))
        md = safe_dict(tab.get("metadata"))
        table_data = safe_dict(md.get("table_data"))
        if not table_data:
            malformed_tables.append({"node_id": nid, "reason": "table_data missing"})
            continue
        columns = safe_list(table_data.get("columns"))
        rows = safe_list(table_data.get("rows"))
        grid = safe_list(table_data.get("grid"))
        if not columns and not rows and not grid:
            malformed_tables.append({"node_id": nid, "reason": "parsed table_data is empty"})
    if malformed_tables:
        report.add(
            "warning",
            "TABLE_DATA_EMPTY",
            "One or more table nodes contain no parsed tabular data",
            examples=malformed_tables[:30],
            count=len(malformed_tables),
        )

    empty_equations = [safe_str(n.get("node_id")) for n in node_types.get("equation", []) if not safe_str(n.get("content")).strip()]
    if empty_equations:
        report.add(
            "warning",
            "EQUATION_CONTENT_EMPTY",
            "Equation nodes with empty LaTeX/content were found",
            node_ids=empty_equations[:50],
            count=len(empty_equations),
        )

    # Stored stats consistency. metadata_structured_nodes may be absent in an old graph,
    # therefore only compare fields present in stats, except total_nodes/total_edges.
    actual_type_counts = Counter(safe_str(n.get("node_type")) for n in nodes)
    stat_mismatches: List[Dict[str, Any]] = []
    for node_type, stat_key in EXPECTED_STATS_BY_TYPE.items():
        if stat_key in stats:
            actual = int(actual_type_counts.get(node_type, 0))
            try:
                stored = int(stats.get(stat_key, 0))
            except Exception:
                stored = stats.get(stat_key)
            if stored != actual:
                stat_mismatches.append({"stat": stat_key, "stored": stored, "actual": actual})
    for stat_key, actual in (("total_nodes", len(nodes)), ("total_edges", len(edges))):
        if stat_key in stats and stats.get(stat_key) != actual:
            stat_mismatches.append({"stat": stat_key, "stored": stats.get(stat_key), "actual": actual})
    if stat_mismatches:
        report.add(
            "error",
            "GRAPH_STATS_MISMATCH",
            "Stored graph statistics do not match graph contents",
            examples=stat_mismatches,
            count=len(stat_mismatches),
        )

    observed_token_counts = [
        int(safe_dict(n.get("metadata")).get("token_count"))
        for n in chunks
        if isinstance(safe_dict(n.get("metadata")).get("token_count"), int)
    ]
    if observed_token_counts and "max_chunk_tokens_observed" in stats:
        actual_max = max(observed_token_counts)
        if stats.get("max_chunk_tokens_observed") != actual_max:
            report.add(
                "warning",
                "GRAPH_TOKEN_STAT_MISMATCH",
                "stats.max_chunk_tokens_observed differs from chunk metadata",
                stored=stats.get("max_chunk_tokens_observed"),
                actual=actual_max,
            )

    graph_summary = {
        "doc_id": doc_id,
        "source_file": source_file,
        "schema_version": graph.get("schema_version"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(sorted(actual_type_counts.items())),
        "chunking_config": chunking,
        "all_node_ids_unique": not bool(dup_node_ids),
        "all_evidence_uids_unique": not bool(dup_uids),
        "all_edge_endpoints_resolve": not bool(dangling),
        "asset_roots_checked": [str(x) for x in asset_roots],
    }
    report.checks["graph"] = graph_summary
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "nodes": nodes,
        "edges": edges,
        "node_by_id": node_by_id,
        "uid_to_node": uid_to_node,
        "node_types": node_types,
        "chunking_config": chunking,
    }


# -----------------------------------------------------------------------------
# retrieval_docs checks
# -----------------------------------------------------------------------------


def validate_retrieval_docs(
    rows: List[Dict[str, Any]],
    graph_ctx: Dict[str, Any],
    report: ValidationReport,
) -> Dict[str, Any]:
    doc_id = graph_ctx["doc_id"]
    source_file = graph_ctx["source_file"]
    node_by_id: Dict[str, Dict[str, Any]] = graph_ctx["node_by_id"]
    chunk_nodes = {
        nid: node
        for nid, node in node_by_id.items()
        if safe_str(node.get("node_type")) == "section_chunk"
    }

    row_ids = [safe_str(r.get("id")).strip() or safe_str(r.get("node_id")).strip() for r in rows]
    row_uids = [safe_str(r.get("evidence_uid")).strip() for r in rows]

    dup_ids = duplicates(row_ids)
    dup_uids = duplicates(row_uids)
    if dup_ids:
        report.add("error", "RETRIEVAL_ID_DUPLICATE", "Duplicate retrieval document IDs detected", values=dup_ids[:50], count=len(dup_ids))
    if dup_uids:
        report.add("error", "RETRIEVAL_UID_DUPLICATE", "Duplicate retrieval evidence_uids detected", values=dup_uids[:50], count=len(dup_uids))

    retrieval_id_set = {x for x in row_ids if x}
    chunk_id_set = set(chunk_nodes)
    missing_rows = sorted(chunk_id_set - retrieval_id_set)
    extra_rows = sorted(retrieval_id_set - chunk_id_set)
    if missing_rows:
        report.add(
            "error",
            "RETRIEVAL_CHUNK_MISSING",
            "section_chunk graph nodes are missing from retrieval_docs.jsonl",
            node_ids=missing_rows[:50],
            count=len(missing_rows),
        )
    if extra_rows:
        report.add(
            "error",
            "RETRIEVAL_ORPHAN_ROW",
            "retrieval_docs.jsonl contains rows that do not resolve to section_chunk graph nodes",
            node_ids=extra_rows[:50],
            count=len(extra_rows),
        )

    row_errors: List[Dict[str, Any]] = []
    text_mismatches: List[str] = []
    token_mismatches: List[Dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()

    for row, nid, uid in zip(rows, row_ids, row_uids):
        if not nid:
            row_errors.append({"node_id": "", "reason": "missing id"})
            continue
        node = chunk_nodes.get(nid)
        if node is None:
            continue

        expected = expected_uid(doc_id, nid) if doc_id else safe_str(node.get("evidence_uid"))
        if not uid or (expected and uid != expected):
            row_errors.append({"node_id": nid, "reason": "evidence_uid mismatch", "value": uid, "expected": expected})
        if safe_str(row.get("node_type")) != "section_chunk":
            row_errors.append({"node_id": nid, "reason": "node_type mismatch", "value": safe_str(row.get("node_type"))})
        if doc_id and safe_str(row.get("doc_id")).strip() != doc_id:
            row_errors.append({"node_id": nid, "reason": "doc_id mismatch", "value": safe_str(row.get("doc_id")), "expected": doc_id})
        if source_file and safe_str(row.get("source_file")).strip() != source_file:
            row_errors.append({"node_id": nid, "reason": "source_file mismatch", "value": safe_str(row.get("source_file")), "expected": source_file})

        row_text = normalize_text_for_compare(row.get("text"))
        node_text = normalize_text_for_compare(node.get("content"))
        if row_text != node_text:
            text_mismatches.append(nid)

        row_md = safe_dict(row.get("metadata"))
        node_md = safe_dict(node.get("metadata"))
        if row_md.get("token_count") != node_md.get("token_count"):
            token_mismatches.append(
                {
                    "node_id": nid,
                    "retrieval": row_md.get("token_count"),
                    "graph": node_md.get("token_count"),
                }
            )

        bucket = normalize_section_bucket(safe_str(row.get("section_title")), safe_str(row.get("section_path")))
        bucket_counts[bucket] += 1

    if row_errors:
        report.add(
            "error",
            "RETRIEVAL_ROW_INCONSISTENT",
            "Retrieval document metadata is inconsistent with paper_graph.json",
            examples=row_errors[:40],
            count=len(row_errors),
        )
    if text_mismatches:
        report.add(
            "error",
            "RETRIEVAL_TEXT_MISMATCH",
            "retrieval_docs.jsonl text differs from the corresponding section_chunk node content",
            node_ids=text_mismatches[:50],
            count=len(text_mismatches),
        )
    if token_mismatches:
        report.add(
            "error",
            "RETRIEVAL_TOKEN_COUNT_MISMATCH",
            "Stored token_count differs between retrieval_docs.jsonl and graph chunk metadata",
            examples=token_mismatches[:40],
            count=len(token_mismatches),
        )

    summary = {
        "row_count": len(rows),
        "graph_chunk_count": len(chunk_nodes),
        "complete_chunk_coverage": not missing_rows and not extra_rows,
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in SECTION_BUCKETS},
    }
    report.checks["retrieval_docs"] = summary
    return summary


# -----------------------------------------------------------------------------
# FAISS/manifest checks
# -----------------------------------------------------------------------------


def validate_faiss(
    output_dir: Path,
    graph_ctx: Dict[str, Any],
    retrieval_summary: Dict[str, Any],
    report: ValidationReport,
    index_dir: Optional[Path] = None,
    deep_faiss: bool = False,
) -> Dict[str, Any]:
    index_dir = (index_dir or (output_dir / "faiss_section_indexes")).resolve()
    manifest_path = index_dir / "faiss_index_manifest.json"

    if not index_dir.exists():
        report.add(
            "warning",
            "FAISS_DIRECTORY_MISSING",
            "FAISS index directory is absent; indexing may have been intentionally skipped",
            path=str(index_dir),
        )
        summary = {"present": False, "index_dir": str(index_dir)}
        report.checks["faiss"] = summary
        return summary

    if not manifest_path.exists():
        report.add(
            "error",
            "FAISS_MANIFEST_MISSING",
            "FAISS directory exists but faiss_index_manifest.json is missing",
            path=str(manifest_path),
        )
        manifest: Dict[str, Any] = {}
    else:
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            report.add("error", "FAISS_MANIFEST_INVALID", "Could not parse FAISS manifest", path=str(manifest_path), error=str(exc))
            manifest = {}

    doc_id = graph_ctx["doc_id"]
    source_file = graph_ctx["source_file"]
    if manifest:
        if doc_id and safe_str(manifest.get("doc_id")) != doc_id:
            report.add("error", "FAISS_DOC_ID_MISMATCH", "FAISS manifest doc_id differs from paper graph", manifest=manifest.get("doc_id"), graph=doc_id)
        if source_file and safe_str(manifest.get("source_file")) != source_file:
            report.add("error", "FAISS_SOURCE_FILE_MISMATCH", "FAISS manifest source_file differs from paper graph", manifest=manifest.get("source_file"), graph=source_file)

        graph_chunking = safe_dict(graph_ctx.get("chunking_config"))
        manifest_chunking = safe_dict(manifest.get("chunking_config_from_graph"))
        if graph_chunking and manifest_chunking and graph_chunking != manifest_chunking:
            report.add(
                "error",
                "FAISS_CHUNK_CONFIG_MISMATCH",
                "FAISS manifest was built from a different chunking configuration",
                graph=graph_chunking,
                manifest=manifest_chunking,
            )

        distance_semantics = safe_str(safe_dict(manifest.get("faiss")).get("distance_semantics")).lower()
        if distance_semantics and "smaller" not in distance_semantics and "lower" not in distance_semantics:
            report.add(
                "warning",
                "FAISS_DISTANCE_SEMANTICS_UNCLEAR",
                "Manifest does not clearly state that smaller L2 distance is better",
                value=safe_dict(manifest.get("faiss")).get("distance_semantics"),
            )

    expected_counts = dict(retrieval_summary.get("bucket_counts", {}))
    metadata_count = 0
    metadata_nodes = graph_ctx["node_types"].get("metadata_structured", [])
    if any(safe_str(n.get("content")).strip() for n in metadata_nodes):
        metadata_count = len([n for n in metadata_nodes if safe_str(n.get("content")).strip()])
    expected_counts["metadata"] = metadata_count

    index_reports = safe_dict(safe_dict(manifest.get("faiss")).get("indexes")) if manifest else {}
    count_mismatches: List[Dict[str, Any]] = []
    file_errors: List[Dict[str, Any]] = []
    deep_results: Dict[str, Any] = {}

    faiss_module = None
    if deep_faiss:
        try:
            import faiss as faiss_module  # type: ignore
        except Exception as exc:
            report.add(
                "warning",
                "FAISS_DEEP_CHECK_UNAVAILABLE",
                "--deep_faiss was requested but the faiss Python package could not be imported",
                error=str(exc),
            )

    for bucket, index_name in INDEX_NAMES.items():
        expected_count = int(expected_counts.get(bucket, 0))
        faiss_file = index_dir / f"{index_name}.faiss"
        pkl_file = index_dir / f"{index_name}.pkl"
        report_entry = safe_dict(index_reports.get(bucket))

        if report_entry:
            stored_count = report_entry.get("document_count")
            if stored_count != expected_count:
                count_mismatches.append(
                    {"bucket": bucket, "manifest_count": stored_count, "expected_count": expected_count}
                )

        if expected_count > 0:
            if not faiss_file.exists() or not pkl_file.exists():
                file_errors.append(
                    {
                        "bucket": bucket,
                        "expected_count": expected_count,
                        "faiss_exists": faiss_file.exists(),
                        "pkl_exists": pkl_file.exists(),
                    }
                )
        else:
            if faiss_file.exists() or pkl_file.exists():
                file_errors.append(
                    {
                        "bucket": bucket,
                        "expected_count": 0,
                        "reason": "stale index files exist for an empty bucket",
                        "faiss_exists": faiss_file.exists(),
                        "pkl_exists": pkl_file.exists(),
                    }
                )

        if faiss_module is not None and faiss_file.exists():
            try:
                idx = faiss_module.read_index(str(faiss_file))
                item = {
                    "ntotal": int(getattr(idx, "ntotal", -1)),
                    "dimension": int(getattr(idx, "d", -1)),
                    "metric_type": int(getattr(idx, "metric_type", -1)),
                }
                deep_results[bucket] = item
                if item["ntotal"] != expected_count:
                    count_mismatches.append(
                        {"bucket": bucket, "faiss_ntotal": item["ntotal"], "expected_count": expected_count}
                    )
                if report_entry.get("embedding_dimension") not in (None, "") and item["dimension"] != report_entry.get("embedding_dimension"):
                    count_mismatches.append(
                        {
                            "bucket": bucket,
                            "faiss_dimension": item["dimension"],
                            "manifest_dimension": report_entry.get("embedding_dimension"),
                        }
                    )
            except Exception as exc:
                report.add(
                    "error",
                    "FAISS_BINARY_READ_FAILED",
                    "Could not inspect a FAISS binary index",
                    bucket=bucket,
                    path=str(faiss_file),
                    error=str(exc),
                )

    if count_mismatches:
        report.add(
            "error",
            "FAISS_COUNT_MISMATCH",
            "FAISS manifest/binary document counts do not match the current retrieval evidence",
            examples=count_mismatches[:40],
            count=len(count_mismatches),
        )
    if file_errors:
        report.add(
            "error",
            "FAISS_FILE_INCONSISTENT",
            "Missing or stale FAISS index file pairs were detected",
            examples=file_errors[:40],
            count=len(file_errors),
        )

    summary = {
        "present": True,
        "index_dir": str(index_dir),
        "manifest_present": manifest_path.exists(),
        "expected_document_counts": expected_counts,
        "deep_faiss_checked": faiss_module is not None,
        "deep_index_stats": deep_results,
    }
    report.checks["faiss"] = summary
    return summary


# -----------------------------------------------------------------------------
# Optional exact tokenizer check
# -----------------------------------------------------------------------------


def validate_tokens_deep(
    graph_ctx: Dict[str, Any],
    report: ValidationReport,
    tokenizer_model: Optional[str],
) -> Dict[str, Any]:
    chunking = safe_dict(graph_ctx.get("chunking_config"))
    model_name = tokenizer_model or safe_str(chunking.get("tokenizer_model")).strip()
    max_chunk_tokens = chunking.get("max_chunk_tokens")
    if not model_name:
        report.add("warning", "TOKENIZER_MODEL_UNKNOWN", "Deep token check requested but no tokenizer model is available")
        summary = {"performed": False}
        report.checks["deep_token_check"] = summary
        return summary

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        report.add(
            "warning",
            "TOKENIZER_CHECK_UNAVAILABLE",
            "Deep token check requested but transformers could not be imported",
            error=str(exc),
        )
        summary = {"performed": False, "model": model_name}
        report.checks["deep_token_check"] = summary
        return summary

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        report.add(
            "error",
            "TOKENIZER_LOAD_FAILED",
            "Could not load tokenizer for deep token validation",
            model=model_name,
            error=str(exc),
        )
        summary = {"performed": False, "model": model_name}
        report.checks["deep_token_check"] = summary
        return summary

    mismatches: List[Dict[str, Any]] = []
    over_limit: List[Dict[str, Any]] = []
    observed: List[int] = []
    for node in graph_ctx["node_types"].get("section_chunk", []):
        nid = safe_str(node.get("node_id"))
        text = safe_str(node.get("content"))
        actual = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        observed.append(actual)
        stored = safe_dict(node.get("metadata")).get("token_count")
        if stored is not None and stored != actual:
            mismatches.append({"node_id": nid, "stored": stored, "actual": actual})
        if isinstance(max_chunk_tokens, int) and actual > max_chunk_tokens:
            over_limit.append({"node_id": nid, "actual": actual, "max_chunk_tokens": max_chunk_tokens})

    metadata_over: List[Dict[str, Any]] = []
    for node in graph_ctx["node_types"].get("metadata_structured", []):
        nid = safe_str(node.get("node_id"))
        text = safe_str(node.get("content"))
        if not text:
            continue
        actual = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        observed.append(actual)
        md_cfg = safe_dict(safe_dict(node.get("metadata")).get("embedding_config"))
        declared = md_cfg.get("max_embedding_tokens")
        if isinstance(declared, int) and actual > declared:
            metadata_over.append({"node_id": nid, "actual": actual, "max_embedding_tokens": declared})

    if mismatches:
        report.add(
            "error",
            "TOKEN_COUNT_RECOMPUTE_MISMATCH",
            "Actual tokenizer counts differ from stored chunk token_count values",
            examples=mismatches[:40],
            count=len(mismatches),
        )
    if over_limit:
        report.add(
            "error",
            "TOKEN_COUNT_RECOMPUTE_OVER_LIMIT",
            "Actual tokenizer counts exceed the configured chunk limit",
            examples=over_limit[:40],
            count=len(over_limit),
        )
    if metadata_over:
        report.add(
            "error",
            "METADATA_TOKEN_OVER_LIMIT",
            "Structured metadata embedding text exceeds its declared token limit",
            examples=metadata_over[:20],
            count=len(metadata_over),
        )

    summary = {
        "performed": True,
        "model": model_name,
        "checked_chunks": len(graph_ctx["node_types"].get("section_chunk", [])),
        "checked_metadata_nodes": len(graph_ctx["node_types"].get("metadata_structured", [])),
        "max_tokens_observed": max(observed, default=0),
        "stored_count_mismatches": len(mismatches),
        "over_limit": len(over_limit) + len(metadata_over),
    }
    report.checks["deep_token_check"] = summary
    return summary


# -----------------------------------------------------------------------------
# Optional generated answer/result validation
# -----------------------------------------------------------------------------


def validate_result_json(
    result: Dict[str, Any],
    graph_ctx: Dict[str, Any],
    report: ValidationReport,
    result_path: Path,
) -> Dict[str, Any]:
    uid_to_node: Dict[str, Dict[str, Any]] = graph_ctx["uid_to_node"]
    unresolved_supports: List[Dict[str, str]] = []
    unresolved_render: List[Dict[str, str]] = []
    wrong_render_types: List[Dict[str, str]] = []

    claims = safe_list(result.get("claims"))
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        cid = safe_str(claim.get("claim_id"))
        # Current final result can contain fully assembled SupportRef objects or raw uids.
        raw_supports = claim.get("support_evidence_uids")
        if isinstance(raw_supports, list):
            support_uids = [safe_str(x) for x in raw_supports]
        else:
            support_uids = []
            for support in safe_list(claim.get("supports")):
                if isinstance(support, dict):
                    support_uids.append(safe_str(support.get("evidence_uid")))
        for uid in support_uids:
            if uid and uid not in uid_to_node:
                unresolved_supports.append({"claim_id": cid, "evidence_uid": uid})

    render_ids = safe_dict(result.get("render_ids"))
    if not render_ids:
        render_ids = safe_dict(result.get("render_payload_ids"))
    for key, expected_type in RENDER_KEY_TO_TYPE.items():
        for uid in safe_list(render_ids.get(key)):
            uid_s = safe_str(uid)
            node = uid_to_node.get(uid_s)
            if node is None:
                unresolved_render.append({"render_key": key, "evidence_uid": uid_s})
            elif safe_str(node.get("node_type")) != expected_type:
                wrong_render_types.append(
                    {
                        "render_key": key,
                        "evidence_uid": uid_s,
                        "node_type": safe_str(node.get("node_type")),
                        "expected_type": expected_type,
                    }
                )

    render_payload = safe_dict(result.get("render_payload"))
    for key, expected_type in RENDER_KEY_TO_TYPE.items():
        for item in safe_list(render_payload.get(key)):
            if not isinstance(item, dict):
                continue
            uid = safe_str(item.get("evidence_uid"))
            node = uid_to_node.get(uid)
            if uid and node is None:
                unresolved_render.append({"render_key": f"render_payload.{key}", "evidence_uid": uid})
            elif node is not None and safe_str(node.get("node_type")) != expected_type:
                wrong_render_types.append(
                    {
                        "render_key": f"render_payload.{key}",
                        "evidence_uid": uid,
                        "node_type": safe_str(node.get("node_type")),
                        "expected_type": expected_type,
                    }
                )

    if unresolved_supports:
        report.add(
            "error",
            "RESULT_SUPPORT_UID_UNRESOLVED",
            "Generated claim supports contain evidence_uids that do not resolve to the graph",
            examples=unresolved_supports[:40],
            count=len(unresolved_supports),
        )
    if unresolved_render:
        report.add(
            "error",
            "RESULT_RENDER_UID_UNRESOLVED",
            "Generated render IDs/payload contain evidence_uids that do not resolve to the graph",
            examples=unresolved_render[:40],
            count=len(unresolved_render),
        )
    if wrong_render_types:
        report.add(
            "error",
            "RESULT_RENDER_TYPE_MISMATCH",
            "Generated render IDs resolve to the wrong evidence object type",
            examples=wrong_render_types[:40],
            count=len(wrong_render_types),
        )

    summary = {
        "path": str(result_path),
        "claim_count": len(claims),
        "unresolved_supports": len(unresolved_supports),
        "unresolved_render_ids": len(unresolved_render),
        "wrong_render_types": len(wrong_render_types),
    }
    report.checks["result_json"] = summary
    return summary


# -----------------------------------------------------------------------------
# Source-document presence check
# -----------------------------------------------------------------------------


def validate_source_presence(
    output_dir: Path,
    graph_ctx: Dict[str, Any],
    report: ValidationReport,
    source_root: Optional[Path] = None,
) -> None:
    source_file = safe_str(graph_ctx.get("source_file")).strip()
    if not source_file or is_remote_uri(source_file):
        return
    p = Path(source_file)
    if p.is_absolute() and p.exists():
        report.checks["source_document"] = {"found": True, "path": str(p)}
        return

    roots: List[Path] = []
    if source_root is not None:
        roots.append(source_root.resolve())
    roots.extend(
        [
            output_dir.resolve(),
            output_dir.parent.resolve(),
            (output_dir.parent / "source").resolve(),
        ]
    )
    candidates = [(r / p).resolve() for r in roots]
    found = next((c for c in candidates if c.exists() and c.is_file()), None)
    if found is None:
        report.add(
            "warning",
            "SOURCE_DOCUMENT_NOT_FOUND",
            "Graph source_file could not be resolved locally; provenance label remains valid but the demo is not fully self-contained",
            source_file=source_file,
            tried=[str(x) for x in candidates],
        )
        report.checks["source_document"] = {"found": False, "source_file": source_file}
    else:
        report.checks["source_document"] = {"found": True, "path": str(found)}


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def validate_pipeline(
    output_dir: str | Path,
    graph_path: Optional[str | Path] = None,
    retrieval_path: Optional[str | Path] = None,
    index_dir: Optional[str | Path] = None,
    asset_base_dir: Optional[str | Path] = None,
    source_root: Optional[str | Path] = None,
    result_json: Optional[str | Path] = None,
    deep_faiss: bool = False,
    deep_token_check: bool = False,
    tokenizer_model: Optional[str] = None,
    strict_warnings: bool = False,
) -> ValidationReport:
    output_dir = Path(output_dir).resolve()
    graph_path_p = Path(graph_path).resolve() if graph_path else output_dir / "paper_graph.json"
    retrieval_path_p = Path(retrieval_path).resolve() if retrieval_path else output_dir / "retrieval_docs.jsonl"
    index_dir_p = Path(index_dir).resolve() if index_dir else output_dir / "faiss_section_indexes"
    asset_base_p = Path(asset_base_dir).resolve() if asset_base_dir else None
    source_root_p = Path(source_root).resolve() if source_root else None

    report = ValidationReport()
    report.checks["inputs"] = {
        "output_dir": str(output_dir),
        "paper_graph": str(graph_path_p),
        "retrieval_docs": str(retrieval_path_p),
        "faiss_index_dir": str(index_dir_p),
        "asset_base_dir": str(asset_base_p) if asset_base_p else None,
        "source_root": str(source_root_p) if source_root_p else None,
        "result_json": str(Path(result_json).resolve()) if result_json else None,
    }

    if not graph_path_p.exists():
        report.add("error", "GRAPH_FILE_MISSING", "paper_graph.json does not exist", path=str(graph_path_p))
        return report

    try:
        graph = read_json(graph_path_p)
    except Exception as exc:
        report.add("error", "GRAPH_FILE_INVALID", "Could not parse paper_graph.json", path=str(graph_path_p), error=str(exc))
        return report

    graph_ctx = validate_graph(
        graph=graph,
        report=report,
        graph_path=graph_path_p,
        asset_base_dir=asset_base_p,
    )

    validate_source_presence(
        output_dir=output_dir,
        graph_ctx=graph_ctx,
        report=report,
        source_root=source_root_p,
    )

    if retrieval_path_p.exists():
        try:
            rows = read_jsonl(retrieval_path_p)
            retrieval_summary = validate_retrieval_docs(rows, graph_ctx, report)
        except Exception as exc:
            report.add("error", "RETRIEVAL_FILE_INVALID", "Could not parse/validate retrieval_docs.jsonl", path=str(retrieval_path_p), error=str(exc))
            retrieval_summary = {"bucket_counts": {bucket: 0 for bucket in SECTION_BUCKETS}}
    else:
        report.add("error", "RETRIEVAL_FILE_MISSING", "retrieval_docs.jsonl does not exist", path=str(retrieval_path_p))
        retrieval_summary = {"bucket_counts": {bucket: 0 for bucket in SECTION_BUCKETS}}

    validate_faiss(
        output_dir=output_dir,
        graph_ctx=graph_ctx,
        retrieval_summary=retrieval_summary,
        report=report,
        index_dir=index_dir_p,
        deep_faiss=deep_faiss,
    )

    if deep_token_check:
        validate_tokens_deep(graph_ctx, report, tokenizer_model=tokenizer_model)

    if result_json:
        result_path = Path(result_json).resolve()
        if not result_path.exists():
            report.add("error", "RESULT_JSON_MISSING", "Requested result JSON does not exist", path=str(result_path))
        else:
            try:
                result = read_json(result_path)
                validate_result_json(result, graph_ctx, report, result_path)
            except Exception as exc:
                report.add("error", "RESULT_JSON_INVALID", "Could not parse/validate result JSON", path=str(result_path), error=str(exc))

    if strict_warnings and report.warnings > 0:
        report.add(
            "error",
            "STRICT_WARNING_FAILURE",
            "--strict_warnings is enabled and one or more warnings were emitted",
            warning_count=report.warnings,
        )

    return report


# -----------------------------------------------------------------------------
# Console formatting / CLI
# -----------------------------------------------------------------------------


def print_report(report: ValidationReport, verbose: bool = False) -> None:
    print("=" * 88)
    print("MatSci-RAG PIPELINE VALIDATION")
    print("=" * 88)
    print(f"Status   : {'PASS' if report.passed else 'FAIL'}")
    print(f"Errors   : {report.errors}")
    print(f"Warnings : {report.warnings}")
    print(f"Info     : {report.infos}")

    if report.findings:
        print("\nFindings")
        print("-" * 88)
        for finding in report.findings:
            tag = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[finding.severity]
            print(f"[{tag}] {finding.code}: {finding.message}")
            if verbose and finding.context:
                print(json.dumps(finding.context, ensure_ascii=False, indent=2))
    else:
        print("\nNo inconsistencies detected.")

    print("\nKey checks")
    print("-" * 88)
    graph_check = safe_dict(report.checks.get("graph"))
    retrieval_check = safe_dict(report.checks.get("retrieval_docs"))
    faiss_check = safe_dict(report.checks.get("faiss"))
    if graph_check:
        print(f"Graph nodes / edges : {graph_check.get('node_count', 'N/A')} / {graph_check.get('edge_count', 'N/A')}")
        print(f"Unique node IDs     : {graph_check.get('all_node_ids_unique', 'N/A')}")
        print(f"Unique evidence UIDs: {graph_check.get('all_evidence_uids_unique', 'N/A')}")
        print(f"Resolved edge ends  : {graph_check.get('all_edge_endpoints_resolve', 'N/A')}")
    if retrieval_check:
        print(f"Retrieval rows       : {retrieval_check.get('row_count', 'N/A')}")
        print(f"Chunk coverage       : {retrieval_check.get('complete_chunk_coverage', 'N/A')}")
    if faiss_check:
        print(f"FAISS present        : {faiss_check.get('present', False)}")
        print(f"FAISS manifest       : {faiss_check.get('manifest_present', False)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MatSci-RAG graph, retrieval documents, evidence identifiers, "
            "cross-modal references, local figure assets, and FAISS build outputs."
        )
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory containing paper_graph.json and retrieval_docs.jsonl.",
    )
    parser.add_argument("--graph", default=None, help="Optional explicit paper_graph.json path.")
    parser.add_argument("--retrieval_docs", default=None, help="Optional explicit retrieval_docs.jsonl path.")
    parser.add_argument("--index_dir", default=None, help="Optional explicit FAISS index directory.")
    parser.add_argument(
        "--asset_base_dir",
        default=None,
        help=(
            "Optional base directory for resolving relative figure image paths. "
            "The validator also tries output_dir and nearby workspace directories."
        ),
    )
    parser.add_argument(
        "--source_root",
        default=None,
        help="Optional root directory for resolving graph source_file (for a self-contained demo check).",
    )
    parser.add_argument(
        "--result_json",
        default=None,
        help="Optional saved main_pipeline result JSON whose claim/render evidence_uids should be validated.",
    )
    parser.add_argument(
        "--deep_faiss",
        action="store_true",
        help="If faiss is installed, inspect binary index ntotal/dimension/metric_type without loading embeddings.",
    )
    parser.add_argument(
        "--deep_token_check",
        action="store_true",
        help="Recompute chunk token counts with the configured Hugging Face tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_model",
        default=None,
        help="Tokenizer path/name for --deep_token_check; defaults to paper_graph.chunking_config.tokenizer_model.",
    )
    parser.add_argument(
        "--strict_warnings",
        action="store_true",
        help="Treat any warning as a validation failure.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output validation report JSON. Default: <output_dir>/validation_report.json",
    )
    parser.add_argument("--verbose", action="store_true", help="Print finding contexts in the console.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        report = validate_pipeline(
            output_dir=args.output_dir,
            graph_path=args.graph,
            retrieval_path=args.retrieval_docs,
            index_dir=args.index_dir,
            asset_base_dir=args.asset_base_dir,
            source_root=args.source_root,
            result_json=args.result_json,
            deep_faiss=args.deep_faiss,
            deep_token_check=args.deep_token_check,
            tokenizer_model=args.tokenizer_model,
            strict_warnings=args.strict_warnings,
        )
    except Exception as exc:
        print(f"[FATAL] Validation could not run: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve() if args.report else output_dir / "validation_report.json"
    write_json(report_path, report.to_dict())
    print_report(report, verbose=args.verbose)
    print(f"\n[REPORT] {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
