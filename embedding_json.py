# -*- coding: utf-8 -*-
from __future__ import annotations

"""
embedding_json.py
=================
Build section-bucketed FAISS indexes for MatSci-RAG.

Expected upstream workflow
--------------------------
1. clean_md.py
2. md_json.py
3. json_split.py
      -> paper_graph.json
      -> retrieval_docs.jsonl
4. metadata_processor.py + merge_metadata_to_graph.py
      -> paper_graph.json containing metadata_structured
5. embedding_json.py
      -> faiss_section_indexes/
           faiss_intro.faiss/.pkl
           faiss_method.faiss/.pkl
           ...
           faiss_metadata.faiss/.pkl
      -> faiss_index_manifest.json

Design decisions
----------------
- No machine-specific paths or document-specific names are hard-coded.
- Section chunks and structured metadata are embedded with the same BGE model.
- Embeddings are L2-normalized. LangChain FAISS therefore uses IndexFlatL2;
  smaller returned distances indicate greater semantic similarity.
- `evidence_uid` is preserved in FAISS metadata as the interface-level evidence ID.
- Chunk lengths are validated against the actual embedding tokenizer before index
  construction so over-length inputs fail explicitly instead of being silently
  truncated by the embedding model.
- The metadata index is stored in the same FAISS directory as section indexes to
  match the current MatSci-RAG loader interface.

Example
-------
python embedding_json.py \
    --output_dir examples/test_case/output \
    --model BAAI/bge-large-en-v1.5 \
    --device cuda

A local model directory can be supplied instead:

python embedding_json.py \
    --output_dir examples/test_case/output \
    --model "D:/BGE_large_en_1.5v"
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_INDEX_DIRNAME = "faiss_section_indexes"
DEFAULT_RETRIEVAL_FILENAME = "retrieval_docs.jsonl"
DEFAULT_GRAPH_FILENAME = "paper_graph.json"

SECTION_BUCKETS = (
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


# ==========================================================
# 0. Basic IO / helpers
# ==========================================================

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_num}: {exc}"
                ) from exc
            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def load_json(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def make_evidence_uid(doc_id: Any, node_id: Any, existing: Any = None) -> str:
    """Return an existing interface ID or construct ``doc_id::node_id``."""
    existing_text = safe_str(existing).strip()
    if existing_text:
        return existing_text

    doc = safe_str(doc_id).strip()
    node = safe_str(node_id).strip()
    if doc and node:
        return f"{doc}::{node}"
    return ""


# ==========================================================
# 1. Section-title normalization
# ==========================================================

def normalize_section_bucket(section_title: str, section_path: str = "") -> str:
    title = (section_title or "").strip().lower()
    path = (section_path or "").strip().lower()
    full = " ".join(f"{path} {title}".split())

    results_and_discussion_patterns = (
        "results and discussion",
        "results & discussion",
        "discussion and results",
        "discussion & results",
        "results discussion",
    )
    if any(pattern in full for pattern in results_and_discussion_patterns):
        return "results_discussion"

    if any(keyword in full for keyword in ("introduction", "background")):
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
    if any(keyword in full for keyword in method_keywords):
        return "method"

    if any(keyword in full for keyword in ("results", "findings")):
        return "results"

    if any(keyword in full for keyword in ("discussion", "general discussion")):
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
    if any(keyword in full for keyword in conclusion_keywords):
        return "conclusion"

    return "other"


# ==========================================================
# 2. Build section-chunk Documents
# ==========================================================

def build_section_documents(
    rows: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, List[Document]]:
    buckets: Dict[str, List[Document]] = {bucket: [] for bucket in SECTION_BUCKETS}

    for item in rows:
        if item.get("node_type") != "section_chunk":
            continue

        text = safe_str(item.get("text")).strip()
        if not text:
            continue

        section_title = safe_str(item.get("section_title"))
        section_path = safe_str(item.get("section_path"))
        bucket = normalize_section_bucket(section_title, section_path)

        md_raw = safe_dict(item.get("metadata"))
        node_id = safe_str(item.get("id")).strip() or safe_str(md_raw.get("chunk_id")).strip()
        doc_id = safe_str(item.get("doc_id")).strip()
        evidence_uid = make_evidence_uid(
            doc_id=doc_id,
            node_id=node_id,
            existing=item.get("evidence_uid") or md_raw.get("evidence_uid"),
        )

        metadata = {
            "doc_id": doc_id,
            "source_file": item.get("source_file"),
            "node_id": node_id,
            "evidence_uid": evidence_uid,
            "node_type": item.get("node_type"),
            "title": item.get("title"),
            "section_id": item.get("section_id"),
            "section_number": item.get("section_number"),
            "section_title": item.get("section_title"),
            "section_title_full": item.get("section_title_full"),
            "section_path": item.get("section_path"),
            "bucket": bucket,
            "chunk_id": md_raw.get("chunk_id"),
            "chunk_number": md_raw.get("chunk_number"),
            "total_chunks": md_raw.get("total_chunks"),
            "global_order": md_raw.get("global_order"),
            "prev_chunk_id": md_raw.get("prev_chunk_id"),
            "next_chunk_id": md_raw.get("next_chunk_id"),
            "token_count": md_raw.get("token_count"),
            "max_chunk_tokens": md_raw.get("max_chunk_tokens"),
            "sentence_overlap": md_raw.get("sentence_overlap"),
            "spacy_model": md_raw.get("spacy_model"),
            "tokenizer_model": md_raw.get("tokenizer_model"),
            "mentioned_figures": safe_list(md_raw.get("mentioned_figures")),
            "mentioned_tables": safe_list(md_raw.get("mentioned_tables")),
            "mentioned_equation_ids": safe_list(md_raw.get("mentioned_equation_ids")),
            "mentioned_equation_numbers": safe_list(md_raw.get("mentioned_equation_numbers")),
            "mentioned_references": safe_list(md_raw.get("mentioned_references")),
            "nearby_evidence_ids": safe_list(md_raw.get("nearby_evidence_ids")),
        }

        buckets[bucket].append(Document(page_content=text, metadata=metadata))

        if verbose:
            sec_label = (
                safe_str(item.get("section_title_full"))
                or safe_str(item.get("section_title"))
                or "UNKNOWN_SECTION"
            )
            print(
                f"[MAP] {sec_label} -> {bucket} | "
                f"{evidence_uid or node_id} | tokens={md_raw.get('token_count', 'N/A')}"
            )

    return buckets


# ==========================================================
# 3. Build metadata_structured Documents
# ==========================================================

def build_metadata_documents_from_graph(
    graph_data: Dict[str, Any],
    verbose: bool = False,
) -> List[Document]:
    docs: List[Document] = []

    for node in safe_list(graph_data.get("nodes")):
        if not isinstance(node, dict) or node.get("node_type") != "metadata_structured":
            continue

        text = safe_str(node.get("content")).strip()
        if not text:
            continue

        md_raw = safe_dict(node.get("metadata"))
        structured_profile = safe_dict(md_raw.get("structured_profile"))
        bib = safe_dict(structured_profile.get("bibliography"))
        abs_profile = safe_dict(structured_profile.get("abstract_profile"))
        ret = safe_dict(structured_profile.get("retrieval_profile"))

        node_id = safe_str(node.get("node_id")).strip() or "metadata_structured"
        doc_id = safe_str(node.get("doc_id")).strip() or safe_str(graph_data.get("doc_id")).strip()
        evidence_uid = make_evidence_uid(
            doc_id=doc_id,
            node_id=node_id,
            existing=node.get("evidence_uid") or md_raw.get("evidence_uid"),
        )

        # Prefer structured_profile values, but gracefully fall back to the
        # flattened metadata fields generated by metadata_processor.py.
        metadata = {
            "doc_id": doc_id,
            "source_file": node.get("source_file") or graph_data.get("source_file"),
            "node_id": node_id,
            "evidence_uid": evidence_uid,
            "node_type": node.get("node_type"),
            "title": bib.get("title", md_raw.get("title", "")),
            "authors": bib.get("authors", md_raw.get("authors", [])),
            "journal": bib.get("journal", md_raw.get("journal", "")),
            "publisher": bib.get("publisher", md_raw.get("publisher", "")),
            "year": bib.get("year", md_raw.get("year")),
            "keywords": bib.get("keywords", md_raw.get("keywords", [])),
            "material_systems": ret.get("material_systems", md_raw.get("material_systems", [])),
            "alloy_family": ret.get("alloy_family", md_raw.get("alloy_family", [])),
            "core_elements": ret.get("core_elements", md_raw.get("core_elements", [])),
            "property_topics": ret.get("property_topics", md_raw.get("property_topics", [])),
            "method_tags": ret.get("method_tags", md_raw.get("method_tags", [])),
            "application_tags": ret.get("application_tags", md_raw.get("application_tags", [])),
            "task_type": ret.get("task_type", md_raw.get("task_type", "")),
            "document_type": ret.get("document_type", md_raw.get("document_type", "")),
            "temperature_context": ret.get("temperature_context", md_raw.get("temperature_context", [])),
            "research_goal": abs_profile.get("research_goal", ""),
            "method_summary": abs_profile.get("method_summary", ""),
            "main_findings": abs_profile.get("main_findings", []),
            "significance": abs_profile.get("significance", ""),
            "bucket": "metadata",
        }

        docs.append(Document(page_content=text, metadata=metadata))

        if verbose:
            print(
                f"[META] {metadata.get('title') or 'UNKNOWN_TITLE'} -> metadata | "
                f"{evidence_uid or node_id}"
            )

    return docs


# ==========================================================
# 4. Embedding model + tokenizer validation
# ==========================================================

def resolve_device(device: str) -> str:
    """Resolve `auto` without making CUDA a hard dependency."""
    requested = (device or "auto").strip().lower()
    if requested != "auto":
        return requested

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_embeddings(
    model_name: str,
    device: str = "auto",
    trust_remote_code: bool = True,
    normalize_embeddings: bool = True,
) -> Tuple[HuggingFaceEmbeddings, str]:
    resolved_device = resolve_device(device)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": resolved_device,
            "trust_remote_code": trust_remote_code,
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
        },
    )
    return embeddings, resolved_device


def load_embedding_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
):
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def get_tokenizer_model_max_length(tokenizer) -> Optional[int]:
    value = getattr(tokenizer, "model_max_length", None)
    if not isinstance(value, int) or value <= 0:
        return None

    # Hugging Face uses very large sentinel values for tokenizers whose maximum
    # length is not explicitly defined. Treat those values as unknown.
    if value > 1_000_000:
        return None
    return int(value)


def count_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded.get("input_ids", []))


def validate_embedding_lengths(
    section_buckets: Dict[str, List[Document]],
    metadata_docs: List[Document],
    tokenizer,
    graph_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate every text before embedding.

    Scientific section chunks are required to fit inside the embedding model's
    sequence length. Metadata is also checked because a long LLM-generated metadata
    profile could otherwise be silently truncated.
    """
    model_max_length = get_tokenizer_model_max_length(tokenizer)
    if model_max_length is None:
        print(
            "[WARN] Embedding tokenizer does not expose a finite model_max_length; "
            "strict overflow validation against the model limit is unavailable."
        )

    checked = 0
    section_checked = 0
    metadata_checked = 0
    maximum_observed = 0
    maximum_label = ""
    mismatched_stored_counts: List[Dict[str, Any]] = []
    overlength: List[Dict[str, Any]] = []
    over_declared_chunk_limit: List[Dict[str, Any]] = []

    chunking_config = safe_dict(graph_data.get("chunking_config"))
    declared_tokenizer = safe_str(chunking_config.get("tokenizer_model")).strip()
    declared_max_chunk_tokens = chunking_config.get("max_chunk_tokens")
    if not isinstance(declared_max_chunk_tokens, int) or declared_max_chunk_tokens <= 0:
        declared_max_chunk_tokens = None

    for bucket, docs in section_buckets.items():
        for doc in docs:
            actual = count_tokens(tokenizer, doc.page_content)
            checked += 1
            section_checked += 1

            label = safe_str(doc.metadata.get("evidence_uid")) or safe_str(doc.metadata.get("node_id"))
            if actual > maximum_observed:
                maximum_observed = actual
                maximum_label = label

            stored = doc.metadata.get("token_count")
            if isinstance(stored, int) and stored != actual:
                mismatched_stored_counts.append(
                    {
                        "evidence_uid": label,
                        "stored_token_count": stored,
                        "embedding_tokenizer_count": actual,
                    }
                )

            if declared_max_chunk_tokens is not None and actual > declared_max_chunk_tokens:
                over_declared_chunk_limit.append(
                    {
                        "evidence_uid": label,
                        "bucket": bucket,
                        "token_count": actual,
                        "declared_max_chunk_tokens": declared_max_chunk_tokens,
                    }
                )

            if model_max_length is not None and actual > model_max_length:
                overlength.append(
                    {
                        "evidence_uid": label,
                        "bucket": bucket,
                        "token_count": actual,
                        "model_max_length": model_max_length,
                    }
                )

    for doc in metadata_docs:
        actual = count_tokens(tokenizer, doc.page_content)
        checked += 1
        metadata_checked += 1
        label = safe_str(doc.metadata.get("evidence_uid")) or safe_str(doc.metadata.get("node_id"))

        if actual > maximum_observed:
            maximum_observed = actual
            maximum_label = label

        if model_max_length is not None and actual > model_max_length:
            overlength.append(
                {
                    "evidence_uid": label,
                    "bucket": "metadata",
                    "token_count": actual,
                    "model_max_length": model_max_length,
                }
            )

    report = {
        "documents_checked": checked,
        "section_documents_checked": section_checked,
        "metadata_documents_checked": metadata_checked,
        "embedding_tokenizer_model_max_length": model_max_length,
        "graph_declared_chunk_tokenizer": declared_tokenizer,
        "graph_declared_max_chunk_tokens": declared_max_chunk_tokens,
        "max_tokens_observed": maximum_observed,
        "max_tokens_observed_evidence_uid": maximum_label,
        "stored_token_count_mismatches": len(mismatched_stored_counts),
        "stored_token_count_mismatch_examples": mismatched_stored_counts[:10],
        "over_declared_chunk_limit": len(over_declared_chunk_limit),
        "over_declared_chunk_limit_examples": over_declared_chunk_limit[:10],
        "overlength_documents": len(overlength),
        "overlength_examples": overlength[:10],
    }

    if mismatched_stored_counts:
        print(
            f"[WARN] {len(mismatched_stored_counts)} section chunk(s) have a stored "
            "token_count that differs from the current embedding tokenizer count. "
            "This usually means json_split.py and embedding_json.py are using "
            "different tokenizer configurations."
        )

    if over_declared_chunk_limit:
        examples = ", ".join(
            f"{item['evidence_uid']}={item['token_count']}"
            for item in over_declared_chunk_limit[:5]
        )
        raise ValueError(
            "Section chunk exceeds the max_chunk_tokens declared by json_split.py. "
            "Refusing to build indexes because the preprocessing and embedding "
            f"configurations are inconsistent. Examples: {examples}"
        )

    if overlength:
        examples = ", ".join(
            f"{item['evidence_uid']}={item['token_count']}" for item in overlength[:5]
        )
        raise ValueError(
            "Embedding input exceeds the model/tokenizer sequence limit. "
            "Refusing to build indexes because this would cause silent truncation. "
            f"Examples: {examples}"
        )

    return report


# ==========================================================
# 5. FAISS index construction
# ==========================================================

def remove_existing_index_files(save_dir: Path, index_name: str) -> None:
    for suffix in (".faiss", ".pkl"):
        path = save_dir / f"{index_name}{suffix}"
        if path.exists():
            path.unlink()


def save_faiss_index(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    save_dir: str | Path,
    index_name: str,
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Rebuilding a document should never leave a stale index behind.
    remove_existing_index_files(save_dir, index_name)

    if not documents:
        print(f"[SKIP] {index_name}: no valid documents")
        return {
            "index_name": index_name,
            "document_count": 0,
            "created": False,
        }

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(str(save_dir), index_name)

    index = getattr(vectordb, "index", None)
    result = {
        "index_name": index_name,
        "document_count": len(documents),
        "created": True,
        "index_class": type(index).__name__ if index is not None else "",
        "metric_type": getattr(index, "metric_type", None) if index is not None else None,
        "embedding_dimension": getattr(index, "d", None) if index is not None else None,
    }

    print(
        f"[OK] {index_name}: {len(documents)} documents -> {save_dir} "
        f"({result['index_class']}, dim={result['embedding_dimension']})"
    )
    return result


# ==========================================================
# 6. Main build function
# ==========================================================

def build_indexes(
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    retrieval_jsonl: Optional[str | Path] = None,
    graph_json: Optional[str | Path] = None,
    index_dir: Optional[str | Path] = None,
    trust_remote_code: bool = True,
    normalize_embeddings: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)

    retrieval_path = (
        Path(retrieval_jsonl)
        if retrieval_jsonl is not None
        else output_dir / DEFAULT_RETRIEVAL_FILENAME
    )
    graph_path = (
        Path(graph_json)
        if graph_json is not None
        else output_dir / DEFAULT_GRAPH_FILENAME
    )
    index_path = (
        Path(index_dir)
        if index_dir is not None
        else output_dir / DEFAULT_INDEX_DIRNAME
    )

    if not retrieval_path.exists():
        raise FileNotFoundError(f"retrieval_docs.jsonl not found: {retrieval_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"paper_graph.json not found: {graph_path}")

    rows = load_jsonl(retrieval_path)
    graph_data = load_json(graph_path)

    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] retrieval docs: {retrieval_path} ({len(rows)} rows)")
    print(f"[INFO] graph: {graph_path}")
    print(f"[INFO] index directory: {index_path}")

    section_buckets = build_section_documents(rows, verbose=verbose)
    metadata_docs = build_metadata_documents_from_graph(graph_data, verbose=verbose)

    print("\n[INFO] Section bucket counts:")
    for bucket in SECTION_BUCKETS:
        print(f"  - {bucket}: {len(section_buckets[bucket])}")
    print(f"  - metadata: {len(metadata_docs)}")

    tokenizer = load_embedding_tokenizer(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )
    token_validation = validate_embedding_lengths(
        section_buckets=section_buckets,
        metadata_docs=metadata_docs,
        tokenizer=tokenizer,
        graph_data=graph_data,
    )

    print("\n[INFO] Token validation:")
    print(
        f"  - model_max_length: "
        f"{token_validation['embedding_tokenizer_model_max_length']}"
    )
    print(f"  - max_tokens_observed: {token_validation['max_tokens_observed']}")
    print(
        f"  - stored token-count mismatches: "
        f"{token_validation['stored_token_count_mismatches']}"
    )
    print(
        f"  - over declared chunk limit: "
        f"{token_validation['over_declared_chunk_limit']}"
    )
    print(f"  - over model sequence limit: {token_validation['overlength_documents']}")

    embeddings, resolved_device = build_embeddings(
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        normalize_embeddings=normalize_embeddings,
    )
    print(f"\n[INFO] Embedding model: {model_name}")
    print(f"[INFO] Device: {resolved_device}")
    print(f"[INFO] normalize_embeddings: {normalize_embeddings}")

    index_path.mkdir(parents=True, exist_ok=True)
    index_reports: Dict[str, Dict[str, Any]] = {}

    for bucket in SECTION_BUCKETS:
        index_reports[bucket] = save_faiss_index(
            documents=section_buckets[bucket],
            embeddings=embeddings,
            save_dir=index_path,
            index_name=INDEX_NAMES[bucket],
        )

    index_reports["metadata"] = save_faiss_index(
        documents=metadata_docs,
        embeddings=embeddings,
        save_dir=index_path,
        index_name=INDEX_NAMES["metadata"],
    )

    manifest = {
        "schema_version": "1.0",
        "doc_id": graph_data.get("doc_id"),
        "source_file": graph_data.get("source_file"),
        "inputs": {
            "retrieval_docs": retrieval_path.name,
            "paper_graph": graph_path.name,
        },
        "embedding": {
            "model": model_name,
            "device": resolved_device,
            "normalize_embeddings": normalize_embeddings,
            "trust_remote_code": trust_remote_code,
            "tokenizer_model_max_length": token_validation[
                "embedding_tokenizer_model_max_length"
            ],
        },
        "faiss": {
            "index_directory": index_path.name,
            "distance_semantics": (
                "LangChain FAISS.from_documents default IndexFlatL2; "
                "smaller L2 distance is more similar. With normalized embeddings, "
                "L2 ranking is monotonic with cosine similarity."
            ),
            "indexes": index_reports,
        },
        "chunking_config_from_graph": safe_dict(graph_data.get("chunking_config")),
        "token_validation": token_validation,
    }

    manifest_path = index_path / "faiss_index_manifest.json"
    write_json(manifest_path, manifest)

    print("\n[DONE] FAISS indexes built successfully")
    print(f"[DONE] Index directory: {index_path}")
    print(f"[DONE] Manifest: {manifest_path}")
    return manifest


# ==========================================================
# 7. CLI
# ==========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build section-bucketed and metadata FAISS indexes for MatSci-RAG "
            "from retrieval_docs.jsonl and paper_graph.json."
        )
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory containing retrieval_docs.jsonl and paper_graph.json. "
            "FAISS indexes are written beneath this directory by default."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face embedding model name or local model path "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cuda, cpu, cuda:0, ... (default: auto)",
    )
    parser.add_argument(
        "--retrieval_jsonl",
        default=None,
        help="Optional explicit retrieval_docs.jsonl path.",
    )
    parser.add_argument(
        "--graph_json",
        default=None,
        help="Optional explicit paper_graph.json path.",
    )
    parser.add_argument(
        "--index_dir",
        default=None,
        help=(
            "Optional explicit FAISS output directory. Defaults to "
            f"<output_dir>/{DEFAULT_INDEX_DIRNAME}."
        ),
    )
    parser.add_argument(
        "--no_normalize_embeddings",
        action="store_true",
        help="Disable L2 normalization of embedding vectors (not recommended for manuscript settings).",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code when loading the model/tokenizer.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print document-level bucket mapping information.",
    )
    args = parser.parse_args()

    build_indexes(
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        retrieval_jsonl=args.retrieval_jsonl,
        graph_json=args.graph_json,
        index_dir=args.index_dir,
        trust_remote_code=not args.no_trust_remote_code,
        normalize_embeddings=not args.no_normalize_embeddings,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
