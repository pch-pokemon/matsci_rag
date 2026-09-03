# -*- coding: utf-8 -*-
from __future__ import annotations

"""
metadata_processor.py
=====================

Structured front-matter / abstract enrichment for the MatSci-RAG evidence graph.

Responsibilities
----------------
1. Read ``metadata_block`` from the Markdown-to-JSON output.
2. Use an LLM to extract a structured document profile.
3. Build a retrieval-oriented ``metadata_structured`` evidence node.
4. Use the same BGE tokenizer family as the downstream embedding stage and enforce
   an explicit metadata embedding-text budget (default: 480 tokens including
   special tokens), avoiding silent truncation by the embedding model.
5. Emit the interface-level ``evidence_uid`` from the source of the node.

Design principles
-----------------
- No machine-specific paths are hard coded.
- API credentials are read only from environment variables / an optional env file.
- The full structured profile is preserved even when the compact embedding text is
  token-budgeted.
- Cached metadata can be upgraded to the current node schema without another LLM
  call when its structured profile is available.
"""

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_LLM_MODEL = "glm-4-plus"
DEFAULT_TOKENIZER_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_MAX_EMBEDDING_TOKENS = 480
DEFAULT_API_KEY_ENV_CANDIDATES: Tuple[str, ...] = (
    "ZHIPUAI_API_KEY",
    "ZHIPU_API_KEY",
    "API_KEY",
)


# ==========================================================
# 0. IO
# ==========================================================

def read_json(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ==========================================================
# 1. Basic helpers
# ==========================================================

def safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_semantic_list(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    cleaned = [safe_str(x).strip() for x in items if safe_str(x).strip()]
    return "; ".join(cleaned)


def make_evidence_uid(doc_id: str, node_id: str = "metadata_structured") -> str:
    doc_id = safe_str(doc_id).strip()
    node_id = safe_str(node_id).strip()
    if not doc_id or not node_id:
        return ""
    return f"{doc_id}::{node_id}"


def infer_doc_id_from_path(input_path: str | Path) -> str:
    # Keep this consistent with the generic public-demo behavior: the file stem is
    # used only when the input JSON itself does not provide doc_id.
    return Path(input_path).stem


def extract_metadata_block_from_md_json(data: Dict[str, Any]) -> str:
    return normalize_text(safe_str(data.get("metadata_block")))


def resolve_source_file(data: Dict[str, Any], input_path: Path) -> str:
    # Prefer the source_file already propagated by md_json/json_split so that the
    # metadata node and the evidence graph share the same document provenance.
    return safe_str(data.get("source_file")).strip() or input_path.name


# ==========================================================
# 2. LLM initialization
# ==========================================================

def _load_env_file(env_file_path: Optional[str | Path]) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required for metadata LLM configuration. "
            "Install it with `pip install python-dotenv`."
        ) from exc

    if env_file_path:
        env_path = Path(env_file_path).expanduser()
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")
        load_dotenv(env_path, override=False)
    else:
        # Also permit a normal project-level .env or already-exported variables.
        load_dotenv(override=False)


def resolve_api_key(
    env_file_path: Optional[str | Path] = None,
    env_candidates: Sequence[str] = DEFAULT_API_KEY_ENV_CANDIDATES,
) -> Tuple[str, str]:
    _load_env_file(env_file_path)
    for name in env_candidates:
        value = os.getenv(name)
        if value:
            return value, name
    raise ValueError(
        "No ZhipuAI API key was found. Set one of: "
        + ", ".join(env_candidates)
        + ", optionally via --env_file."
    )


def build_metadata_llm(
    env_file_path: Optional[str | Path] = None,
    model_name: str = DEFAULT_LLM_MODEL,
):
    try:
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required for metadata structuring."
        ) from exc

    api_key, _ = resolve_api_key(env_file_path=env_file_path)
    return ChatZhipuAI(
        temperature=0,
        model=model_name,
        api_key=api_key,
    )


# ==========================================================
# 3. Default schema
# ==========================================================

def build_default_metadata_profile(
    doc_id: str,
    source_file: str,
    raw_metadata_block: str,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "metadata_type": "document_profile",
        "bibliography": {
            "title": "",
            "authors": [],
            "affiliations": [],
            "corresponding_author": "",
            "email": "",
            "journal": "",
            "publisher": "",
            "year": None,
            "article_history": {
                "received": "",
                "revised": "",
                "accepted": "",
                "available_online": "",
            },
            "keywords": [],
        },
        "abstract_profile": {
            "abstract": "",
            "research_goal": "",
            "method_summary": "",
            "main_findings": [],
            "significance": "",
        },
        "retrieval_profile": {
            "material_systems": [],
            "alloy_family": [],
            "core_elements": [],
            "property_topics": [],
            "method_tags": [],
            "application_tags": [],
            "task_type": "",
            "document_type": "",
            "temperature_context": [],
        },
        "raw_metadata_block": raw_metadata_block,
        "parse_status": {
            "llm_called": False,
            "json_parsed": False,
            "fallback_used": True,
        },
        "llm_raw_response": "",
    }


def deep_merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(default)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


# ==========================================================
# 4. LLM output parsing
# ==========================================================

def extract_json_from_llm_text(text: str) -> Dict[str, Any]:
    text = safe_str(text).strip()
    if not text:
        return {}

    cleaned = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


# ==========================================================
# 5. Prompt
# ==========================================================

def build_metadata_prompt(doc_id: str, source_file: str, metadata_block: str) -> str:
    return f"""
You are an information extraction assistant for scientific paper front matter.
Your task is to convert the metadata block before the Introduction section into a structured JSON object.

Rules:
1. Extract only from the provided text. Do not invent facts.
2. If a field is not explicitly available, use "" for strings, [] for lists, and null for year.
3. Preserve the original scientific meaning of the abstract.
4. Keep research_goal, method_summary, and significance concise.
5. main_findings must be a list of short statements.
6. material_systems, alloy_family, core_elements, property_topics, method_tags, and application_tags must be retrieval-oriented tags, not full sentences.
7. document_type should be something like "research article", "review article", or "" if unknown.
8. Output JSON only. No markdown, no explanation.

Return exactly this JSON schema:

{{
  "doc_id": "{doc_id}",
  "source_file": "{source_file}",
  "metadata_type": "document_profile",
  "bibliography": {{
    "title": "",
    "authors": [],
    "affiliations": [],
    "corresponding_author": "",
    "email": "",
    "journal": "",
    "publisher": "",
    "year": null,
    "article_history": {{
      "received": "",
      "revised": "",
      "accepted": "",
      "available_online": ""
    }},
    "keywords": []
  }},
  "abstract_profile": {{
    "abstract": "",
    "research_goal": "",
    "method_summary": "",
    "main_findings": [],
    "significance": ""
  }},
  "retrieval_profile": {{
    "material_systems": [],
    "alloy_family": [],
    "core_elements": [],
    "property_topics": [],
    "method_tags": [],
    "application_tags": [],
    "task_type": "",
    "document_type": "",
    "temperature_context": []
  }}
}}

Metadata block:
{metadata_block}
""".strip()


# ==========================================================
# 6. LLM call + structuring
# ==========================================================

def call_llm_for_metadata_structuring(
    metadata_block: str,
    doc_id: str,
    source_file: str,
    llm,
) -> Dict[str, Any]:
    metadata_block = normalize_text(metadata_block)
    default_profile = build_default_metadata_profile(
        doc_id=doc_id,
        source_file=source_file,
        raw_metadata_block=metadata_block,
    )

    if not metadata_block:
        return default_profile

    prompt = build_metadata_prompt(doc_id, source_file, metadata_block)

    try:
        response = llm.invoke(prompt)
        raw_text = getattr(response, "content", None) or str(response)
        parsed = extract_json_from_llm_text(raw_text)

        if not parsed:
            result = copy.deepcopy(default_profile)
            result["parse_status"] = {
                "llm_called": True,
                "json_parsed": False,
                "fallback_used": True,
            }
            result["llm_raw_response"] = raw_text
            return result

        merged = deep_merge_dict(default_profile, parsed)
        # Provenance is controlled by the local pipeline, not by model output.
        merged["doc_id"] = doc_id
        merged["source_file"] = source_file
        merged["metadata_type"] = "document_profile"
        merged["raw_metadata_block"] = metadata_block
        merged["parse_status"] = {
            "llm_called": True,
            "json_parsed": True,
            "fallback_used": False,
        }
        merged["llm_raw_response"] = raw_text
        return merged

    except Exception as exc:
        result = copy.deepcopy(default_profile)
        result["parse_status"] = {
            "llm_called": True,
            "json_parsed": False,
            "fallback_used": True,
            "error": str(exc),
        }
        return result


# ==========================================================
# 7. Token-budgeted metadata embedding text
# ==========================================================

def load_tokenizer(tokenizer_model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to enforce the metadata embedding-token budget."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    raw_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(raw_max, int) and 0 < raw_max < 1_000_000:
        model_max_length: Optional[int] = raw_max
    else:
        model_max_length = None
    return tokenizer, model_max_length


def count_tokens(tokenizer, text: str, add_special_tokens: bool = True) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=add_special_tokens, truncation=False))


def _truncate_text_to_total_budget(tokenizer, text: str, max_total_tokens: int) -> str:
    """Truncate a single text string so its encoded length including special tokens fits."""
    if count_tokens(tokenizer, text, add_special_tokens=True) <= max_total_tokens:
        return text

    special_count = count_tokens(tokenizer, "", add_special_tokens=True)
    content_budget = max_total_tokens - special_count
    if content_budget <= 0:
        raise ValueError("max_total_tokens is too small for tokenizer special tokens.")

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)[:content_budget]
    truncated = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    # Defensive tightening for tokenizers whose decode/re-encode is not perfectly stable.
    while truncated and count_tokens(tokenizer, truncated, add_special_tokens=True) > max_total_tokens:
        token_ids = token_ids[:-1]
        truncated = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
    return truncated


def _metadata_embedding_candidates(profile: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return retrieval-oriented fields in priority order.

    The full abstract/profile remains in ``structured_profile``. The embedding text is
    intentionally compact and prioritizes high-signal retrieval descriptors before the
    potentially long abstract, preventing hidden BGE truncation.
    """
    bib = safe_dict(profile.get("bibliography"))
    abs_p = safe_dict(profile.get("abstract_profile"))
    ret = safe_dict(profile.get("retrieval_profile"))

    return [
        ("Title", safe_str(bib.get("title"))),
        ("Keywords", join_semantic_list(bib.get("keywords"))),
        ("Material systems", join_semantic_list(ret.get("material_systems"))),
        ("Alloy family", join_semantic_list(ret.get("alloy_family"))),
        ("Core elements", join_semantic_list(ret.get("core_elements"))),
        ("Property topics", join_semantic_list(ret.get("property_topics"))),
        ("Method tags", join_semantic_list(ret.get("method_tags"))),
        ("Application tags", join_semantic_list(ret.get("application_tags"))),
        ("Task type", safe_str(ret.get("task_type"))),
        ("Document type", safe_str(ret.get("document_type"))),
        ("Temperature context", join_semantic_list(ret.get("temperature_context"))),
        ("Research goal", safe_str(abs_p.get("research_goal"))),
        ("Method summary", safe_str(abs_p.get("method_summary"))),
        ("Main findings", join_semantic_list(abs_p.get("main_findings"))),
        ("Significance", safe_str(abs_p.get("significance"))),
        # Abstract is useful but deliberately lowest priority because it may be long
        # and largely duplicates the concise fields above.
        ("Abstract", safe_str(abs_p.get("abstract"))),
    ]


def build_metadata_embedding_text(
    profile: Dict[str, Any],
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    max_embedding_tokens: int = DEFAULT_MAX_EMBEDDING_TOKENS,
) -> Tuple[str, Dict[str, Any]]:
    if max_embedding_tokens <= 0:
        raise ValueError("max_embedding_tokens must be > 0")

    tokenizer, model_max_length = load_tokenizer(tokenizer_model)
    if model_max_length is not None and max_embedding_tokens > model_max_length:
        raise ValueError(
            f"max_embedding_tokens={max_embedding_tokens} exceeds tokenizer "
            f"model_max_length={model_max_length}."
        )

    selected_lines: List[str] = []
    included_fields: List[str] = []
    truncated_field: Optional[str] = None
    omitted_fields: List[str] = []

    candidates = [(label, normalize_text(value)) for label, value in _metadata_embedding_candidates(profile) if normalize_text(value)]

    for idx, (label, value) in enumerate(candidates):
        line = f"{label}: {value}"
        candidate_text = "\n".join(selected_lines + [line])
        if count_tokens(tokenizer, candidate_text, add_special_tokens=True) <= max_embedding_tokens:
            selected_lines.append(line)
            included_fields.append(label)
            continue

        # Use any remaining budget for a partial version of this field, then stop.
        prefix = "\n".join(selected_lines)
        if prefix:
            separator = "\n"
            base = prefix + separator + f"{label}: "
        else:
            base = f"{label}: "

        if count_tokens(tokenizer, base, add_special_tokens=True) < max_embedding_tokens:
            # Binary search the value token length to safely fill the remaining budget.
            value_ids = tokenizer.encode(value, add_special_tokens=False, truncation=False)
            lo, hi = 0, len(value_ids)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                partial_value = tokenizer.decode(
                    value_ids[:mid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                trial_line = f"{label}: {partial_value}" if partial_value else f"{label}:"
                trial_text = "\n".join(selected_lines + [trial_line])
                if count_tokens(tokenizer, trial_text, add_special_tokens=True) <= max_embedding_tokens:
                    best = trial_line
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best and best not in {f"{label}:", f"{label}: "}:
                selected_lines.append(best)
                included_fields.append(label)
                truncated_field = label

        omitted_fields.extend([candidate_label for candidate_label, _ in candidates[idx + 1 :]])
        if label not in included_fields:
            omitted_fields.insert(0, label)
        break

    embedding_text = "\n".join(selected_lines).strip()
    if not embedding_text:
        # This is mainly a guard for pathological tokenizer/budget combinations.
        title = safe_str(safe_dict(profile.get("bibliography")).get("title")).strip()
        embedding_text = _truncate_text_to_total_budget(
            tokenizer,
            f"Title: {title}" if title else "Structured scientific document metadata",
            max_embedding_tokens,
        )

    token_count = count_tokens(tokenizer, embedding_text, add_special_tokens=True)
    if token_count > max_embedding_tokens:
        raise RuntimeError(
            f"Metadata embedding token budget violated: {token_count} > {max_embedding_tokens}"
        )

    report = {
        "tokenizer_model": tokenizer_model,
        "tokenizer_model_max_length": model_max_length,
        "max_embedding_tokens": max_embedding_tokens,
        "token_count": token_count,
        "token_count_includes_special_tokens": True,
        "included_fields": included_fields,
        "truncated_field": truncated_field,
        "omitted_fields": omitted_fields,
    }
    return embedding_text, report


# ==========================================================
# 8. Build metadata node
# ==========================================================

def build_metadata_node(
    profile: Dict[str, Any],
    doc_id: str,
    source_file: str,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    max_embedding_tokens: int = DEFAULT_MAX_EMBEDDING_TOKENS,
) -> Dict[str, Any]:
    bib = safe_dict(profile.get("bibliography"))
    ret = safe_dict(profile.get("retrieval_profile"))
    embedding_text, token_report = build_metadata_embedding_text(
        profile=profile,
        tokenizer_model=tokenizer_model,
        max_embedding_tokens=max_embedding_tokens,
    )

    node_id = "metadata_structured"
    evidence_uid = make_evidence_uid(doc_id, node_id)

    return {
        "node_id": node_id,
        "evidence_uid": evidence_uid,
        "node_type": "metadata_structured",
        "doc_id": doc_id,
        "source_file": source_file,
        "title": safe_str(bib.get("title")) or "Structured document metadata",
        "content": embedding_text,
        "raw_content": json.dumps(profile, ensure_ascii=False),
        "section_id": None,
        "section_number": "",
        "section_title": "",
        "section_title_full": "",
        "parent_section_id": None,
        "hierarchy": None,
        "section_path": "",
        "order_index": None,
        "metadata": {
            "block_type": "front_matter_structured",
            "evidence_uid": evidence_uid,
            "title": bib.get("title", ""),
            "authors": bib.get("authors", []),
            "affiliations": bib.get("affiliations", []),
            "corresponding_author": bib.get("corresponding_author", ""),
            "email": bib.get("email", ""),
            "journal": bib.get("journal", ""),
            "publisher": bib.get("publisher", ""),
            "year": bib.get("year", None),
            "article_history": bib.get("article_history", {}),
            "keywords": bib.get("keywords", []),
            "material_systems": ret.get("material_systems", []),
            "alloy_family": ret.get("alloy_family", []),
            "core_elements": ret.get("core_elements", []),
            "property_topics": ret.get("property_topics", []),
            "method_tags": ret.get("method_tags", []),
            "application_tags": ret.get("application_tags", []),
            "task_type": ret.get("task_type", ""),
            "document_type": ret.get("document_type", ""),
            "temperature_context": ret.get("temperature_context", []),
            "embedding_config": token_report,
            "structured_profile": profile,
        },
    }


# ==========================================================
# 9. Cache upgrade / validation
# ==========================================================

def _profile_from_cached_node(node: Dict[str, Any]) -> Dict[str, Any]:
    metadata = safe_dict(node.get("metadata"))
    profile = safe_dict(metadata.get("structured_profile"))
    if profile:
        return profile

    raw_content = safe_str(node.get("raw_content")).strip()
    if raw_content:
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict) and parsed.get("metadata_type") == "document_profile":
                return parsed
        except Exception:
            pass
    return {}


def try_upgrade_cached_node(
    cached_node: Dict[str, Any],
    doc_id: str,
    source_file: str,
    tokenizer_model: str,
    max_embedding_tokens: int,
) -> Optional[Dict[str, Any]]:
    if safe_str(cached_node.get("node_type")) != "metadata_structured":
        return None
    cached_doc_id = safe_str(cached_node.get("doc_id")).strip()
    if cached_doc_id and cached_doc_id != doc_id:
        return None

    profile = _profile_from_cached_node(cached_node)
    if not profile:
        return None

    profile = copy.deepcopy(profile)
    profile["doc_id"] = doc_id
    profile["source_file"] = source_file
    return build_metadata_node(
        profile=profile,
        doc_id=doc_id,
        source_file=source_file,
        tokenizer_model=tokenizer_model,
        max_embedding_tokens=max_embedding_tokens,
    )


# ==========================================================
# 10. Main process
# ==========================================================

def process_metadata(
    input_json_path: str,
    output_dir: str,
    env_file_path: Optional[str] = None,
    model_name: str = DEFAULT_LLM_MODEL,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    max_embedding_tokens: int = DEFAULT_MAX_EMBEDDING_TOKENS,
    doc_id: Optional[str] = None,
    debug: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    input_path = Path(input_json_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata_node_path = outdir / "metadata_node.json"
    metadata_structured_path = outdir / "metadata_structured.json"
    metadata_llm_raw_path = outdir / "metadata_llm_raw.txt"
    metadata_embedding_text_path = outdir / "metadata_embedding_text.txt"
    metadata_summary_path = outdir / "metadata_process_summary.json"

    data = read_json(input_path)
    resolved_doc_id = safe_str(doc_id).strip() or safe_str(data.get("doc_id")).strip() or infer_doc_id_from_path(input_path)
    source_file = resolve_source_file(data, input_path)
    metadata_block = extract_metadata_block_from_md_json(data)

    if not metadata_block:
        raise ValueError("metadata_block is empty in input JSON.")

    # Cache path: upgrade old metadata nodes to the current evidence_uid/token-budget
    # schema without a second LLM call whenever structured_profile is available.
    if metadata_node_path.exists() and not force:
        cached_node = read_json(metadata_node_path)
        upgraded = try_upgrade_cached_node(
            cached_node=cached_node,
            doc_id=resolved_doc_id,
            source_file=source_file,
            tokenizer_model=tokenizer_model,
            max_embedding_tokens=max_embedding_tokens,
        )
        if upgraded is not None:
            write_json(metadata_node_path, upgraded)
            profile = _profile_from_cached_node(upgraded)
            if profile:
                write_json(metadata_structured_path, profile)
            token_report = safe_dict(safe_dict(upgraded.get("metadata")).get("embedding_config"))
            summary = {
                "doc_id": resolved_doc_id,
                "source_file": source_file,
                "evidence_uid": upgraded.get("evidence_uid"),
                "cache_hit": True,
                "cache_upgraded_to_current_schema": True,
                "llm_called": False,
                "embedding_token_count": token_report.get("token_count"),
                "max_embedding_tokens": token_report.get("max_embedding_tokens"),
                "tokenizer_model": token_report.get("tokenizer_model"),
                "outputs": {
                    "metadata_node.json": str(metadata_node_path),
                    "metadata_structured.json": str(metadata_structured_path),
                },
            }
            write_json(metadata_summary_path, summary)
            return summary

    llm = build_metadata_llm(env_file_path=env_file_path, model_name=model_name)
    profile = call_llm_for_metadata_structuring(
        metadata_block=metadata_block,
        doc_id=resolved_doc_id,
        source_file=source_file,
        llm=llm,
    )

    node = build_metadata_node(
        profile=profile,
        doc_id=resolved_doc_id,
        source_file=source_file,
        tokenizer_model=tokenizer_model,
        max_embedding_tokens=max_embedding_tokens,
    )

    # The structured profile is useful reproducibility material, so save it by
    # default. The raw model response remains debug-only.
    write_json(metadata_structured_path, profile)
    write_json(metadata_node_path, node)

    if debug:
        write_text(metadata_llm_raw_path, safe_str(profile.get("llm_raw_response")))
        write_text(metadata_embedding_text_path, safe_str(node.get("content")))

    token_report = safe_dict(safe_dict(node.get("metadata")).get("embedding_config"))
    parse_status = safe_dict(profile.get("parse_status"))
    summary: Dict[str, Any] = {
        "doc_id": resolved_doc_id,
        "source_file": source_file,
        "evidence_uid": node.get("evidence_uid"),
        "cache_hit": False,
        "metadata_block_length_chars": len(metadata_block),
        "embedding_text_length_chars": len(safe_str(node.get("content"))),
        "embedding_token_count": token_report.get("token_count"),
        "max_embedding_tokens": token_report.get("max_embedding_tokens"),
        "tokenizer_model": token_report.get("tokenizer_model"),
        "llm_model": model_name,
        "llm_called": parse_status.get("llm_called", False),
        "json_parsed": parse_status.get("json_parsed", False),
        "fallback_used": parse_status.get("fallback_used", True),
        "outputs": {
            "metadata_node.json": str(metadata_node_path),
            "metadata_structured.json": str(metadata_structured_path),
        },
    }

    if debug:
        summary["outputs"]["metadata_llm_raw.txt"] = str(metadata_llm_raw_path)
        summary["outputs"]["metadata_embedding_text.txt"] = str(metadata_embedding_text_path)

    write_json(metadata_summary_path, summary)
    summary["outputs"]["metadata_process_summary.json"] = str(metadata_summary_path)
    return summary


# ==========================================================
# 11. CLI
# ==========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure scientific-paper metadata and build metadata_structured evidence."
    )
    parser.add_argument("--input", required=True, help="Markdown-to-JSON output file.")
    parser.add_argument("--outdir", required=True, help="Directory for metadata outputs.")
    parser.add_argument("--doc_id", default=None, help="Optional document identifier override.")
    parser.add_argument(
        "--env_file",
        default=None,
        help="Optional .env file containing ZHIPUAI_API_KEY / ZHIPU_API_KEY / API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Metadata-structuring LLM model name.")
    parser.add_argument(
        "--tokenizer_model",
        default=DEFAULT_TOKENIZER_MODEL,
        help="Tokenizer used to enforce the metadata embedding-text budget.",
    )
    parser.add_argument(
        "--max_embedding_tokens",
        type=int,
        default=DEFAULT_MAX_EMBEDDING_TOKENS,
        help="Maximum metadata embedding-text tokens including special tokens (default: 480).",
    )
    parser.add_argument("--debug", action="store_true", help="Save raw LLM and embedding-text debug files.")
    parser.add_argument("--force", action="store_true", help="Ignore cache and run metadata structuring again.")
    args = parser.parse_args()

    summary = process_metadata(
        input_json_path=args.input,
        output_dir=args.outdir,
        env_file_path=args.env_file,
        model_name=args.model,
        tokenizer_model=args.tokenizer_model,
        max_embedding_tokens=args.max_embedding_tokens,
        doc_id=args.doc_id,
        debug=args.debug,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
