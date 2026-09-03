from __future__ import annotations

"""
main_pipeline.py
================
Reference implementation of the graph-aware, evidence-grounded multimodal MatSci-RAG pipeline used for the revised study.

设计目标
--------
1. 串起完整主流程：
   Step 1 Query Understanding
   Step 2 Bucketed Retrieval
   Step 3 Graph Evidence Expansion
   Step 4 LLM Answer Generation (minimal output)
   Step 5 Program Assembler
   Step 6 Final JSON Packaging

2. 先保证“结构正确、接口清晰、可逐步替换细节模块”
3. 不强依赖某一个具体模型或 API，后续可以替换成自己的 LLM / reranker / vision 模块

当前版本特点
------------
- 优先面向单篇或多篇 paper_graph.json + faiss_section_indexes 组织方式
- 图 / 表 / 公式 / 参考文献由程序侧组装，不要求 LLM 生成完整对象
- 一套共享 RAG backbone 支持 extraction / QA 两种显式 task mode
"""

import json
import re
import base64
import mimetypes
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
import os


@dataclass
class PipelineConfig:
    """Runtime configuration for the public MatSci-RAG pipeline.

    The defaults mirror the controlled manuscript configuration:
    - BGE-large-en-v1.5 embeddings
    - BGE-reranker-large reranking
    - initial retrieval top-k = 20
    - reranked top-n = 10
    - at most 20 evidence objects before final budgeting
    - at most 5,120 textual evidence tokens
    - at most 5 linked figure/table objects
    - deterministic generation (temperature = 0)
    """

    output_dir: str
    embedding_model_path: str = "BAAI/bge-large-en-v1.5"
    rerank_model_path: str = "BAAI/bge-reranker-large"
    env_file_path: str = ""
    llm_model_name: str = "GLM-4.1V-Thinking-Flash"
    device: str = "cuda"
    llm_temperature: float = 0.0

    retrieval_top_k: int = 20
    rerank_top_n: int = 10
    metadata_top_k: int = 3

    max_evidence_objects: int = 20
    max_textual_evidence_tokens: int = 5120
    max_visual_tabular_items: int = 5

    # Framework-component switches. Defaults reproduce the full MatSci-RAG model.
    # ``enable_structure=False`` removes section-level localization and searches
    # all section buckets. ``enable_association=False`` disables explicit
    # graph-based expansion from retrieved text to linked evidence objects.
    enable_structure: bool = True
    enable_association: bool = True

    # Use the embedding tokenizer to measure the evidence-context budget unless
    # a dedicated tokenizer is explicitly supplied.
    context_tokenizer_path: str = ""

    # When enabled, locally available associated figure images are attached to
    # the multimodal LLM request in addition to their captions. If the adapter
    # rejects multimodal messages, generation automatically falls back to the
    # text-only evidence prompt.
    enable_multimodal: bool = True
    asset_base_dir: str = ""

    # Research/reproducibility mode: generation failures are raised by default
    # instead of being silently replaced by a synthetic fallback answer.
    allow_generation_fallback: bool = False

    faiss_dirname: str = "faiss_section_indexes"
    graph_filename: str = "paper_graph.json"
    retrieval_docs_filename: str = "retrieval_docs.jsonl"

    index_names: Dict[str, str] = field(default_factory=lambda: {
        "intro": "faiss_intro",
        "method": "faiss_method",
        "results": "faiss_results",
        "discussion": "faiss_discussion",
        "results_discussion": "faiss_results_discussion",
        "conclusion": "faiss_conclusion",
        "other": "faiss_other",
        "metadata": "faiss_metadata",
    })

# ============================================================
# 0. 基础工具
# ============================================================
def build_llm(config: PipelineConfig):
    from dotenv import load_dotenv
    from langchain_community.chat_models import ChatZhipuAI

    if config.env_file_path:
        load_dotenv(config.env_file_path)
    api_key = (
        os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("ZHIPU_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No ZhipuAI API key found. Set ZHIPUAI_API_KEY (recommended), "
            "ZHIPU_API_KEY, or legacy API_KEY."
        )

    llm = ChatZhipuAI(
        temperature=config.llm_temperature,
        model=config.llm_model_name,
        api_key=api_key,
    )
    return llm

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

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = safe_str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_query_analysis_prompt(query: str) -> str:
    return f"""
You are a query-understanding module for a scientific paper QA pipeline.

Your job is to analyze the user question and output a JSON object.

Available query_type values:
- fact
- method
- mechanism
- comparison
- design

Available retrieval_buckets values:
- intro
- method
- results
- discussion
- results_discussion
- conclusion
- other

Rules:
1. Output JSON only.
2. Do not include markdown fences.
3. normalized_query should preserve the user meaning, but make wording cleaner if needed.
4. query_type should reflect the main intent:
   - method: asks how something was designed, prepared, measured, processed, heat treated, tested
   - mechanism: asks why / mechanism / reason / interpretation
   - comparison: asks difference / compare / versus
   - design: asks optimization / screening / alloy design strategy
   - fact: all other factual questions
5. retrieval_buckets should be the most relevant section buckets for answering the question.
6. needs_decomposition is true only if the question clearly contains multiple separable sub-questions.
7. support flags should indicate whether figures/tables/equations/references are likely needed to answer well.
8. vision_required should usually be false unless image-level visual interpretation is truly necessary.

Return exactly this schema:
{{
  "normalized_query": "",
  "query_type": "fact",
  "needs_decomposition": false,
  "sub_questions": [],
  "retrieval_buckets": [],
  "need_figures": false,
  "need_tables": false,
  "need_equations": false,
  "need_references": false,
  "vision_required": false,
  "vision_reason": ""
}}

User question:
{query}
""".strip()

def extract_query_analysis_from_llm(raw_text: str, original_query: str) -> Optional[QueryAnalysis]:
    data = extract_json_from_llm_text(raw_text)
    if not data:
        return None

    allowed_query_types = {"fact", "method", "mechanism", "comparison", "design"}
    allowed_buckets = {"intro", "method", "results", "discussion", "results_discussion", "conclusion", "other"}

    normalized_query = safe_str(data.get("normalized_query")).strip() or normalize_text(original_query)
    query_type = safe_str(data.get("query_type")).strip().lower()
    if query_type not in allowed_query_types:
        query_type = "fact"

    retrieval_buckets_raw = ensure_list(data.get("retrieval_buckets"))
    retrieval_buckets = []
    for b in retrieval_buckets_raw:
        b = safe_str(b).strip().lower()
        if b in allowed_buckets and b not in retrieval_buckets:
            retrieval_buckets.append(b)

    if not retrieval_buckets:
        retrieval_buckets = ["results", "discussion"]

    needs_decomposition = bool(data.get("needs_decomposition", False))
    sub_questions = [safe_str(x).strip() for x in ensure_list(data.get("sub_questions")) if safe_str(x).strip()]

    return QueryAnalysis(
        original_query=original_query,
        normalized_query=normalized_query,
        query_type=query_type,
        needs_decomposition=needs_decomposition,
        sub_questions=sub_questions,
        retrieval_buckets=retrieval_buckets,
        need_figures=bool(data.get("need_figures", False)),
        need_tables=bool(data.get("need_tables", False)),
        need_equations=bool(data.get("need_equations", False)),
        need_references=bool(data.get("need_references", False)),
        vision_required=bool(data.get("vision_required", False)),
        vision_reason=safe_str(data.get("vision_reason")).strip(),
    )

def normalize_task_mode(task_mode: str) -> str:
    """Normalize the benchmark task mode without asking the LLM to infer it."""
    mode = safe_str(task_mode).strip().lower().replace("-", "_")
    aliases = {
        "structured_extraction": "extraction",
        "extract": "extraction",
        "extraction": "extraction",
        "literature_qa": "qa",
        "literature_grounded_qa": "qa",
        "question_answering": "qa",
        "qa": "qa",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported task_mode={task_mode!r}. Use 'extraction' or 'qa'."
        )
    return aliases[mode]


def build_structured_extraction_prompt(
    query: str,
    llm_context: str,
) -> str:
    """Benchmark-aligned structured-extraction prompt from Table S3."""
    return f"""
You are given evidence extracted from a scientific publication. Extract only the requested materials information that is explicitly supported by the provided evidence. Do not infer, estimate, or introduce information that is not stated in the evidence. Preserve reported units and experimental conditions. If a requested field is not reported, return null.

Return the result strictly in the following JSON format:

{{ "composition": ..., "processing": ..., "experimental conditions": ..., "property": ... }}

Example:

Evidence: “The nominal composition of the alloy was Co-9Al-9W (at.%). The alloy was solution-treated at 1200 °C for 12 h and subsequently aged at 900 °C for 100 h. Isothermal oxidation was conducted in air at 900 °C for 100 h, resulting in a mass gain of 7.53 mg cm-2.”

Output:

{{ "composition": {{"Co": "82 at.%", "Al": "9 at.%", "W": "9 at.%"}}, "processing": {{"solution treatment": {{"temperature": "1200 °C", "time": "12 h"}}, "aging": {{"temperature": "900 °C", "time": "100 h"}}}}, "experimental conditions": {{"atmosphere": "air", "temperature": "900 °C", "time": "100 h"}}, "property": {{"mass gain": "7.53 mg cm-2"}} }}

Requested information:
{query}

Evidence:
{llm_context}
""".strip()


def build_literature_grounded_qa_prompt(
    query: str,
    llm_context: str,
) -> str:
    """Benchmark-aligned literature-grounded QA prompt from Table S3."""
    return f"""
You are given evidence extracted from scientific publications. Answer the question using only the provided scientific evidence. Integrate information across the retrieved evidence when necessary, while preserving the experimental conditions and source context associated with each finding. Do not infer, estimate, or introduce information that is not stated in the evidence. When multiple studies report different findings, preserve their corresponding experimental conditions and do not merge them into a single unsupported conclusion. If the available evidence is insufficient to answer the question, explicitly state that the evidence is insufficient.

Question:
{query}

Evidence:
{llm_context}
""".strip()


def build_generation_prompt(
    task_mode: str,
    query: str,
    llm_context: str,
) -> Tuple[str, str]:
    """Return ``(prompt, template_name)`` for the explicit benchmark task."""
    mode = normalize_task_mode(task_mode)
    if mode == "extraction":
        return (
            build_structured_extraction_prompt(query=query, llm_context=llm_context),
            "structured_extraction_v1",
        )
    return (
        build_literature_grounded_qa_prompt(query=query, llm_context=llm_context),
        "literature_grounded_qa_v1",
    )


def normalize_extraction_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce the four-field benchmark schema while accepting minor key variants."""
    if not isinstance(data, dict):
        data = {}

    experimental = data.get("experimental conditions")
    if experimental is None:
        experimental = data.get("experimental_conditions")
    if experimental is None:
        experimental = data.get("experimental condition")

    return {
        "composition": data.get("composition", None),
        "processing": data.get("processing", None),
        "experimental conditions": experimental,
        "property": data.get("property", None),
    }


def build_embeddings(config: PipelineConfig):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_path,
        model_kwargs={
            "device": config.device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

def load_faiss_index(index_dir: Path, index_name: str, embeddings):
    from langchain_community.vectorstores import FAISS

    faiss_file = index_dir / f"{index_name}.faiss"
    pkl_file = index_dir / f"{index_name}.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        return None

    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSONL line {line_no} parse failed: {exc}")
    return rows



def ensure_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []



def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}



def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""



def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def path_to_data_uri(path: Path) -> Optional[str]:
    """Convert a local image file to a data URI for multimodal chat adapters."""
    try:
        if not path.exists() or not path.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"
    except Exception:
        return None


def resolve_image_uri(image_path: str, image_url: str, graph: "EvidenceGraph") -> Optional[str]:
    """Resolve a figure image to an HTTP/data URI or a local file converted to data URI."""
    url = safe_str(image_url).strip()
    if url.startswith(("http://", "https://", "data:image/")):
        return url

    raw = safe_str(image_path).strip() or url
    if not raw:
        return None
    if raw.startswith(("http://", "https://", "data:image/")):
        return raw

    p = Path(raw)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if graph.base_dir is not None:
            candidates.extend([
                graph.base_dir / p,
                graph.base_dir.parent / p,
            ])
        candidates.append(p)

    for candidate in candidates:
        uri = path_to_data_uri(candidate.resolve())
        if uri:
            return uri
    return None

def build_paper_citation(graph: "EvidenceGraph") -> str:
    """
    从 metadata_structured 节点构造本文献引用。
    当前固定格式：
    Title, Year, Journal
    """
    node = graph.get_metadata_node()
    if not node:
        return ""

    md = ensure_dict(node.get("metadata"))

    title = safe_str(md.get("title")).strip()
    journal = safe_str(md.get("journal")).strip()
    year = md.get("year", None)

    parts: List[str] = []
    if title:
        parts.append(title)
    if year not in (None, ""):
        parts.append(str(year))
    if journal:
        parts.append(journal)

    return ", ".join(parts) if parts else ""

# ============================================================
# 1. 数据结构
# ============================================================


@dataclass
class SourceRef:
    doc_id: str
    source_file: str
    corpus_id: Optional[str] = None


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    query_type: str = "fact"
    needs_decomposition: bool = False
    sub_questions: List[str] = field(default_factory=list)
    retrieval_buckets: List[str] = field(default_factory=lambda: ["results", "discussion"])
    need_figures: bool = False
    need_tables: bool = False
    need_equations: bool = False
    need_references: bool = False
    vision_required: bool = False
    vision_reason: str = ""


@dataclass
class RetrievalHit:
    evidence_uid: str
    node_id: str
    node_type: str
    source: SourceRef
    retrieval_distance: float = 0.0
    rerank_score: Optional[float] = None
    bucket: str = ""
    section_id: str = ""
    section_number: str = ""
    section_title: str = ""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Backward-compatible alias. For FAISS IndexFlatL2 this is a distance: lower is better."""
        return self.retrieval_distance


@dataclass
class SupportRef:
    evidence_uid: str
    source: SourceRef
    node_id: str
    node_type: str
    display_label: str = ""
    support_role: str = "primary_text"


@dataclass
class Claim:
    claim_id: str
    text: str
    supports: List[SupportRef] = field(default_factory=list)


@dataclass
class LLMAnswerDraft:
    # Benchmark task output. For QA this is a string; for extraction it is the
    # four-field JSON object defined in Table S3.
    task_mode: str = "qa"
    task_output: Any = None

    # Backward-compatible convenience field for QA. Extraction leaves this empty.
    answer: str = ""
    claims: List[Claim] = field(default_factory=list)
    render_ids: Dict[str, List[str]] = field(default_factory=lambda: {
        "figures": [],
        "tables": [],
        "equations": [],
        "references": [],
    })
    generation_debug: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2. Graph 索引
# ============================================================


class EvidenceGraph:
    """Lightweight accessor for one ``paper_graph.json``.

    ``node_id`` remains document-local. ``evidence_uid`` is the interface-level
    identifier and is resolved explicitly at graph boundaries.
    """

    def __init__(
        self,
        graph_data: Dict[str, Any],
        corpus_id: Optional[str] = None,
        base_dir: Optional[str | Path] = None,
    ):
        self.graph_data = graph_data
        self.doc_id = safe_str(graph_data.get("doc_id"))
        self.source_file = safe_str(graph_data.get("source_file"))
        self.corpus_id = corpus_id
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self.nodes = ensure_list(graph_data.get("nodes"))
        self.edges = ensure_list(graph_data.get("edges"))

        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.node_by_uid: Dict[str, Dict[str, Any]] = {}
        self.out_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.in_edges: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            node_id = safe_str(node.get("node_id"))
            if not node_id:
                continue
            self.node_by_id[node_id] = node
            uid = safe_str(node.get("evidence_uid")) or f"{self.doc_id}::{node_id}"
            self.node_by_uid[uid] = node

        for edge in self.edges:
            src = safe_str(edge.get("source"))
            tgt = safe_str(edge.get("target"))
            self.out_edges.setdefault(src, []).append(edge)
            self.in_edges.setdefault(tgt, []).append(edge)

    def make_source_ref(self) -> SourceRef:
        return SourceRef(
            doc_id=self.doc_id,
            source_file=self.source_file,
            corpus_id=self.corpus_id,
        )

    def make_evidence_uid(self, node_id: str) -> str:
        node = self.get_node(node_id)
        existing = safe_str((node or {}).get("evidence_uid"))
        return existing or f"{self.doc_id}::{node_id}"

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.node_by_id.get(node_id)

    def get_node_by_uid(self, evidence_uid: str) -> Optional[Dict[str, Any]]:
        return self.node_by_uid.get(safe_str(evidence_uid))

    def resolve_evidence_uid(self, evidence_uid: str) -> Optional[Dict[str, Any]]:
        uid = safe_str(evidence_uid).strip()
        if not uid:
            return None
        node = self.get_node_by_uid(uid)
        if node is not None:
            return node
        if "::" not in uid:
            return None
        doc_id, node_id = uid.split("::", 1)
        if doc_id != self.doc_id:
            return None
        return self.get_node(node_id)

    def get_metadata_node(self) -> Optional[Dict[str, Any]]:
        return self.get_node("metadata_structured")

    def get_out_edges(self, node_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        edges = self.out_edges.get(node_id, [])
        if relation is None:
            return edges
        return [e for e in edges if e.get("relation") == relation]

    def get_in_edges(self, node_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        edges = self.in_edges.get(node_id, [])
        if relation is None:
            return edges
        return [e for e in edges if e.get("relation") == relation]


# ============================================================
# 3. Step 1 - Query Understanding
# ============================================================
class LLMQueryAnalyzer:
    def __init__(self, llm, fallback: Optional[QueryAnalyzer] = None):
        self.llm = llm
        self.fallback = fallback or QueryAnalyzer()

    def analyze(self, query: str) -> QueryAnalysis:
        q = normalize_text(query)
        if not q:
            return self.fallback.analyze(query)

        prompt = build_query_analysis_prompt(q)

        try:
            resp = self.llm.invoke(prompt)
            raw_text = getattr(resp, "content", None) or str(resp)

            parsed = extract_query_analysis_from_llm(
                raw_text=raw_text,
                original_query=query,
            )
            if parsed is None:
                raise ValueError("LLM returned invalid query-analysis JSON")

            return parsed

        except Exception as e:
            print(f"[WARN] LLMQueryAnalyzer fallback: {e}")
            return self.fallback.analyze(query)

class QueryAnalyzer:
    """
    第一版先给规则骨架。
    后续你可以把 analyze() 替换成真正的 LLM prompt。
    """

    FIGURE_HINTS = ["fig", "figure", "图", "曲线", "峰", "xrd", "sem", "tem"]
    TABLE_HINTS = ["table", "表", "数据表"]
    EQUATION_HINTS = ["equation", "eq.", "公式"]
    REF_HINTS = ["reference", "文献", "citation", "来源"]

    def analyze(self, query: str) -> QueryAnalysis:
        q = normalize_text(query)
        lower = q.lower()

        query_type = "fact"
        if any(k in lower for k in ["mechanism", "why", "原因", "机理"]):
            query_type = "mechanism"
        elif any(k in lower for k in ["compare", "difference", "相比", "对比"]):
            query_type = "comparison"
        elif any(k in lower for k in ["design", "optimize", "筛选", "设计"]):
            query_type = "design"

        need_figures = any(k in lower for k in self.FIGURE_HINTS)
        need_tables = any(k in lower for k in self.TABLE_HINTS)
        need_equations = any(k in lower for k in self.EQUATION_HINTS)
        need_references = True if any(k in lower for k in self.REF_HINTS) else False

        retrieval_buckets = self._infer_buckets(lower, query_type)
        needs_decomposition, sub_questions = self._infer_decomposition(q)

        vision_required = False
        vision_reason = ""
        if need_figures and any(k in lower for k in ["trend", "curve", "peak", "morphology", "看图", "根据图"]):
            vision_required = False
            vision_reason = "caption_first_default"

        return QueryAnalysis(
            original_query=query,
            normalized_query=q,
            query_type=query_type,
            needs_decomposition=needs_decomposition,
            sub_questions=sub_questions,
            retrieval_buckets=retrieval_buckets,
            need_figures=need_figures,
            need_tables=need_tables,
            need_equations=need_equations,
            need_references=need_references,
            vision_required=vision_required,
            vision_reason=vision_reason,
        )

    def _infer_buckets(self, lower_query: str, query_type: str) -> List[str]:
        if query_type == "mechanism":
            return ["results", "discussion", "results_discussion"]
        if query_type == "comparison":
            return ["results", "discussion"]
        if query_type == "design":
            return ["results", "discussion", "conclusion"]
        if any(k in lower_query for k in ["how", "method", "experiment", "experimental", "制备", "测试"]):
            return ["method", "results"]
        return ["results", "discussion"]

    def _infer_decomposition(self, query: str) -> Tuple[bool, List[str]]:
        # 第一版做非常保守的拆分：只在明显并列问句时拆
        parts = [p.strip() for p in re.split(r"[；;]", query) if p.strip()]
        if len(parts) >= 2:
            return True, parts
        return False, []


# ============================================================
# 4. Step 2 - Retrieval
# ============================================================
class BgeRerank:
    """Cross-encoder reranker. Larger reranker scores are better."""

    def __init__(self, model_path: str, top_n: int = 10, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        self.model_path = model_path
        self.top_n = top_n
        kwargs: Dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = CrossEncoder(self.model_path, max_length=512, **kwargs)

    def rerank_hits(
        self,
        query: str,
        hits: List[RetrievalHit],
        top_n: Optional[int] = None,
    ) -> List[RetrievalHit]:
        if not hits:
            return []

        k = top_n if top_n is not None else self.top_n
        k = max(0, min(int(k), len(hits)))
        if k == 0:
            return []

        pairs = [[query, h.text] for h in hits]
        scores = self.model.predict(pairs)

        ranked_pairs = sorted(
            zip(hits, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:k]

        out: List[RetrievalHit] = []
        for hit, score in ranked_pairs:
            hit.rerank_score = float(score)
            out.append(hit)
        return out


class FaissBucketedRetriever:
    """Structure-aware FAISS retrieval followed by optional BGE reranking.

    The indexes created by ``embedding_json.py`` use normalized embeddings with
    FAISS ``IndexFlatL2``. Therefore ``similarity_search_with_score`` returns an
    L2 *distance*: smaller values are more similar. The original public demo
    sorted those distances in descending order; this implementation intentionally
    fixes that direction.
    """

    def __init__(
        self,
        retrieval_docs: List[Dict[str, Any]],
        bucket_vectorstores: Dict[str, Any],
        metadata_vectorstore: Optional[Any] = None,
        reranker: Optional[BgeRerank] = None,
        default_top_k: int = 20,
        rerank_top_n: int = 10,
    ):
        self.retrieval_docs = retrieval_docs
        self.bucket_vectorstores = bucket_vectorstores
        self.metadata_vectorstore = metadata_vectorstore
        self.reranker = reranker
        self.default_top_k = int(default_top_k)
        self.rerank_top_n = int(rerank_top_n)
        self.last_debug: Dict[str, Any] = {}

        self.row_by_node_id: Dict[str, Dict[str, Any]] = {}
        self.rows_by_bucket: Dict[str, List[Dict[str, Any]]] = {}

        for row in retrieval_docs:
            node_id = safe_str(row.get("id")) or safe_str(row.get("node_id"))
            md = ensure_dict(row.get("metadata"))
            bucket = safe_str(md.get("bucket")) or safe_str(row.get("bucket")) or "other"

            if node_id:
                self.row_by_node_id[node_id] = row
            self.rows_by_bucket.setdefault(bucket, []).append(row)

    def search(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        retrieval_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> List[RetrievalHit]:
        if not query.strip():
            return []

        initial_k = int(retrieval_top_k or self.default_top_k)
        final_n = int(rerank_top_n or self.rerank_top_n)
        initial_k = max(1, initial_k)
        final_n = max(1, min(final_n, initial_k))

        initial_hits = self._search_faiss(
            query=query,
            buckets=buckets,
            graph=graph,
            top_k=initial_k,
        )

        self.last_debug = {
            "distance_strategy": "EUCLIDEAN_DISTANCE",
            "distance_semantics": "lower_is_better",
            "initial_retrieval_top_k": initial_k,
            "initial_candidate_count": len(initial_hits),
            "rerank_top_n": final_n,
            "buckets": list(buckets),
            "initial_candidates": [
                {
                    "evidence_uid": h.evidence_uid,
                    "node_id": h.node_id,
                    "retrieval_distance": h.retrieval_distance,
                    "bucket": h.bucket,
                }
                for h in initial_hits
            ],
        }

        if self.reranker is not None and initial_hits:
            reranked = self.reranker.rerank_hits(
                query=query,
                hits=initial_hits,
                top_n=final_n,
            )
            self.last_debug["reranked_count"] = len(reranked)
            self.last_debug["reranked"] = [
                {
                    "evidence_uid": h.evidence_uid,
                    "node_id": h.node_id,
                    "retrieval_distance": h.retrieval_distance,
                    "rerank_score": h.rerank_score,
                }
                for h in reranked
            ]
            return reranked

        result = initial_hits[:final_n]
        self.last_debug["reranked_count"] = len(result)
        return result

    def _search_faiss(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        top_k: int,
    ) -> List[RetrievalHit]:
        raw_hits: List[RetrievalHit] = []
        seen_node_ids: Set[str] = set()

        # Retrieve a sufficiently broad pool from each structurally localized
        # bucket, merge, deduplicate, then keep the globally best top-k by L2.
        per_bucket_k = max(1, int(top_k))

        for bucket in buckets:
            vs = self.bucket_vectorstores.get(bucket)
            if vs is None:
                continue

            try:
                docs_with_scores = vs.similarity_search_with_score(query, k=per_bucket_k)
            except Exception as exc:
                print(f"[WARN] FAISS search failed for bucket={bucket}: {exc}")
                continue

            for doc, distance in docs_with_scores:
                hit = self._doc_to_hit(
                    doc=doc,
                    distance=distance,
                    bucket=bucket,
                    graph=graph,
                )
                if hit is None or hit.node_id in seen_node_ids:
                    continue
                seen_node_ids.add(hit.node_id)
                raw_hits.append(hit)

        # IndexFlatL2: smaller distance = more similar.
        raw_hits.sort(key=lambda h: h.retrieval_distance)
        return raw_hits[:top_k]

    def _doc_to_hit(
        self,
        doc: Any,
        distance: Any,
        bucket: str,
        graph: EvidenceGraph,
    ) -> Optional[RetrievalHit]:
        md = ensure_dict(getattr(doc, "metadata", {}) or {})
        text = normalize_text(safe_str(getattr(doc, "page_content", "")))

        node_id = (
            safe_str(md.get("node_id"))
            or safe_str(md.get("id"))
            or safe_str(md.get("chunk_id"))
        )
        if not node_id:
            return None

        row = self.row_by_node_id.get(node_id, {})
        row_md = ensure_dict(row.get("metadata"))
        merged_md = {**row_md, **md}

        if graph.get_node(node_id) is None:
            return None

        section_id = safe_str(md.get("section_id")) or safe_str(row.get("section_id"))
        section_number = safe_str(md.get("section_number")) or safe_str(row.get("section_number"))
        section_title = safe_str(md.get("section_title")) or safe_str(row.get("section_title"))
        node_type = safe_str(md.get("node_type")) or safe_str(row.get("node_type")) or "section_chunk"
        hit_bucket = safe_str(md.get("bucket")) or safe_str(row_md.get("bucket")) or bucket

        evidence_uid = (
            safe_str(md.get("evidence_uid"))
            or safe_str(row.get("evidence_uid"))
            or graph.make_evidence_uid(node_id)
        )

        return RetrievalHit(
            evidence_uid=evidence_uid,
            node_id=node_id,
            node_type=node_type,
            source=graph.make_source_ref(),
            retrieval_distance=self._as_float(distance),
            rerank_score=None,
            bucket=hit_bucket,
            section_id=section_id,
            section_number=section_number,
            section_title=section_title,
            text=text,
            metadata=merged_md,
        )

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("inf")

class BucketedRetriever:
    """
    第一版主骨架：
    - 你后续可以接真实 FAISS
    - 当前保留一个 retrieval_docs 的文本匹配兜底实现
    """

    def __init__(self, retrieval_docs: List[Dict[str, Any]]):
        self.retrieval_docs = retrieval_docs
        self.docs_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for row in retrieval_docs:
            md = ensure_dict(row.get("metadata"))
            bucket = safe_str(md.get("bucket")) or safe_str(row.get("bucket")) or "other"
            self.docs_by_bucket.setdefault(bucket, []).append(row)

    def search(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        top_k: int = 8,
    ) -> List[RetrievalHit]:
        candidates: List[Dict[str, Any]] = []
        for bucket in buckets:
            candidates.extend(self.docs_by_bucket.get(bucket, []))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        q_terms = self._simple_terms(query)
        for row in candidates:
            text = normalize_text(safe_str(row.get("text")))
            score = self._simple_score(q_terms, text)
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits: List[RetrievalHit] = []

        for score, row in scored[:top_k]:
            md = ensure_dict(row.get("metadata"))
            node_id = safe_str(row.get("id"))
            hits.append(
                RetrievalHit(
                    evidence_uid=graph.make_evidence_uid(node_id),
                    node_id=node_id,
                    node_type=safe_str(row.get("node_type")) or "section_chunk",
                    source=graph.make_source_ref(),
                    retrieval_distance=-float(score),
                    bucket=safe_str(md.get("bucket")),
                    section_id=safe_str(row.get("section_id")),
                    section_number=safe_str(row.get("section_number")),
                    section_title=safe_str(row.get("section_title")),
                    text=text,
                    metadata=md,
                )
            )
        return hits

    @staticmethod
    def _simple_terms(text: str) -> List[str]:
        text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\-\+]+", " ", text.lower())
        return [w for w in text.split() if len(w) >= 2]

    @staticmethod
    def _simple_score(query_terms: List[str], text: str) -> float:
        lower = text.lower()
        score = 0.0
        for term in query_terms:
            if term in lower:
                score += 1.0
        return score


# ============================================================
# 5. Step 3 - Graph Evidence Expansion
# ============================================================

class GraphExpander:

    def _get_bucket_obj(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        bucket: str,
        node_id: str,
    ) -> Optional[Dict[str, Any]]:
        for obj in result[bucket]:
            if safe_str(obj.get("node_id")) == node_id:
                return obj
        return None

    def expand(
        self,
        hits: List[RetrievalHit],
        graph: EvidenceGraph,
        query_analysis: QueryAnalysis,
        enable_association: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {
            "chunks": [],
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

        seen: Dict[str, set[str]] = {k: set() for k in result}

        for hit in hits:
            node = graph.get_node(hit.node_id)
            if not node:
                continue

            # 0) 主命中 chunk
            self._add_node(
                result,
                seen,
                "chunks",
                node,
                graph,
                triggered_by_chunk_id=hit.node_id,
                expand_role="primary_hit",
            )

            md = ensure_dict(node.get("metadata"))

            # 1) 前一个 chunk
            prev_chunk_id = safe_str(md.get("prev_chunk_id"))
            if prev_chunk_id:
                prev_node = graph.get_node(prev_chunk_id)
                if prev_node:
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        prev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="context_prev",
                    )

            # 2) 后一个 chunk
            next_chunk_id = safe_str(md.get("next_chunk_id"))
            if next_chunk_id:
                next_node = graph.get_node(next_chunk_id)
                if next_node:
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        next_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="context_next",
                    )

            # 3) nearby_evidence_ids
            for ev_id in ensure_list(md.get("nearby_evidence_ids")):
                ev_node = graph.get_node(safe_str(ev_id))
                if not ev_node:
                    continue

                ev_type = safe_str(ev_node.get("node_type"))
                if ev_type == "section_chunk":
                    # Textual neighborhood is retained in every setting. The
                    # association ablation removes explicit text-to-object links,
                    # not ordinary neighboring textual context.
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        ev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="nearby_chunk",
                    )
                elif enable_association:
                    self._route_object_node(
                        result,
                        seen,
                        ev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                    )

            # 4) relation-based object expansion. Disabled for the
            # w/o Association ablation while leaving the textual retrieval
            # backbone and neighboring textual context unchanged.
            if enable_association:
                for edge in graph.get_out_edges(hit.node_id):
                    rel = safe_str(edge.get("relation"))
                    target = safe_str(edge.get("target"))
                    node2 = graph.get_node(target)
                    if not node2:
                        continue

                    if rel in {"cites_figure", "cites_table", "cites_equation", "cites_reference"}:
                        self._route_object_node(
                            result,
                            seen,
                            node2,
                            graph,
                            triggered_by_chunk_id=hit.node_id,
                        )

#        if not query_analysis.need_figures:
#            result["figures"] = []
#        if not query_analysis.need_tables:
#            result["tables"] = []
#        if not query_analysis.need_equations:
#            result["equations"] = []
#        if not query_analysis.need_references:
#            result["references"] = []
# 注意：
# query_analysis.need_figures / need_tables / need_equations / need_references
# 在当前版本中只作为“问题理解信号”，不在 expand 阶段做硬裁剪。
# GraphExpander 的职责是尽量完整地构造证据链；
# 最终是否展示某类对象证据，由后续 LLM render_ids 决定。

        return result

    def _route_object_node(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        seen: Dict[str, set[str]],
        node: Dict[str, Any],
        graph: EvidenceGraph,
        triggered_by_chunk_id: Optional[str] = None,
    ) -> None:

        node_type = safe_str(node.get("node_type"))
        if node_type == "figure":
            self._add_node(result, seen, "figures", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "table":
            self._add_node(result, seen, "tables", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "equation":
            self._add_node(result, seen, "equations", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "reference":
            self._add_node(result, seen, "references", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)


    def _add_node(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        seen: Dict[str, set[str]],
        bucket: str,
        node: Dict[str, Any],
        graph: EvidenceGraph,
        triggered_by_chunk_id: Optional[str] = None,
        expand_role: str = "",
    ) -> None:
        node_id = safe_str(node.get("node_id"))
        if not node_id:
            return

        # 如果已经存在，则只补 triggered_by_chunk_ids；
        # 对 chunk 来说，第一次写入的 expand_role / expanded_from 通常保留即可
        if node_id in seen[bucket]:
            existing = self._get_bucket_obj(result, bucket, node_id)
            if existing and triggered_by_chunk_id:
                refs = existing.setdefault("triggered_by_chunk_ids", [])
                if triggered_by_chunk_id not in refs:
                    refs.append(triggered_by_chunk_id)
            return

        seen[bucket].add(node_id)

        node_copy = dict(node)
        node_copy["evidence_uid"] = graph.make_evidence_uid(node_id)
        node_copy["triggered_by_chunk_ids"] = []

        if triggered_by_chunk_id:
            node_copy["triggered_by_chunk_ids"].append(triggered_by_chunk_id)

        # 只要是进入 expanded_evidence 的 chunk，都补来源标签
        if bucket == "chunks":
            node_copy["expand_role"] = expand_role
            node_copy["expanded_from"] = triggered_by_chunk_id or ""

        result[bucket].append(node_copy)



# ============================================================
# 5.5 Evidence budgeting
# ============================================================

class EvidenceBudgetManager:
    """Apply the manuscript evidence budgets before generation.

    Budget order:
    1. Cap the expanded MatSci-RAG evidence set at ``max_evidence_objects``.
    2. Cap linked figure/table objects at ``max_visual_tabular_items``.
    3. Build the textual evidence context block-by-block without exceeding
       ``max_textual_tokens`` measured by the configured tokenizer.

    Evidence units are never silently truncated mid-unit. If a block would exceed
    the remaining textual budget it is skipped and the next smaller block is tried.
    """

    BUCKETS = ("chunks", "figures", "tables", "equations", "references")

    def __init__(
        self,
        tokenizer_model_path: str,
        max_evidence_objects: int = 20,
        max_textual_tokens: int = 5120,
        max_visual_tabular_items: int = 5,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer_model_path = tokenizer_model_path
        self.max_evidence_objects = max(1, int(max_evidence_objects))
        self.max_textual_tokens = max(1, int(max_textual_tokens))
        self.max_visual_tabular_items = max(0, int(max_visual_tabular_items))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path,
            trust_remote_code=True,
        )

    #def count_tokens(self, text: str) -> int:
    #    if not text:
    #        return 0
    #    # ``tokenize`` is used only for deterministic budget accounting. It does
    #    # not trigger the embedding model's 512-token sequence-length warning
    #    # when measuring a multi-chunk context that may be several thousand
    #    # tokens long.
    #    return len(self.tokenizer.tokenize(text))
    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text or "",
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        return len(encoded["input_ids"])    

    @staticmethod
    def _uid(obj: Dict[str, Any]) -> str:
        return safe_str(obj.get("evidence_uid")).strip()

    def _hit_rank_map(self, hits: List[RetrievalHit]) -> Dict[str, int]:
        return {h.node_id: i for i, h in enumerate(hits)}

    def _trigger_rank(self, obj: Dict[str, Any], hit_rank: Dict[str, int]) -> int:
        ranks = [
            hit_rank[x]
            for x in ensure_list(obj.get("triggered_by_chunk_ids"))
            if x in hit_rank
        ]
        return min(ranks) if ranks else 10**6

    def apply_object_cap(
        self,
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        hits: List[RetrievalHit],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        hit_rank = self._hit_rank_map(hits)
        candidates: List[Tuple[Tuple[int, int, int], str, Dict[str, Any]]] = []

        # Primary reranked chunks are always the highest priority.
        role_priority = {
            "primary_hit": 0,
            "context_prev": 2,
            "context_next": 2,
            "nearby_chunk": 3,
        }

        for position, obj in enumerate(expanded_evidence.get("chunks", [])):
            role = safe_str(obj.get("expand_role"))
            if role == "primary_hit":
                primary_rank = hit_rank.get(safe_str(obj.get("node_id")), 10**6)
                key = (0, primary_rank, position)
            else:
                trigger_rank = hit_rank.get(safe_str(obj.get("expanded_from")), 10**6)
                key = (role_priority.get(role, 3), trigger_rank, position)
            candidates.append((key, "chunks", obj))

        bucket_priority = {
            "figures": 1,
            "tables": 1,
            "equations": 1,
            "references": 4,
        }
        for bucket in ("figures", "tables", "equations", "references"):
            for position, obj in enumerate(expanded_evidence.get(bucket, [])):
                candidates.append((
                    (
                        bucket_priority[bucket],
                        self._trigger_rank(obj, hit_rank),
                        position,
                    ),
                    bucket,
                    obj,
                ))

        candidates.sort(key=lambda x: x[0])

        selected: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.BUCKETS}
        selected_uids: Set[str] = set()
        visual_tabular_count = 0

        for _, bucket, obj in candidates:
            if sum(len(selected[k]) for k in self.BUCKETS) >= self.max_evidence_objects:
                break

            uid = self._uid(obj)
            if not uid or uid in selected_uids:
                continue

            if bucket in {"figures", "tables"}:
                if visual_tabular_count >= self.max_visual_tabular_items:
                    continue
                visual_tabular_count += 1

            selected[bucket].append(obj)
            selected_uids.add(uid)

        report = {
            "max_evidence_objects": self.max_evidence_objects,
            "max_visual_tabular_items": self.max_visual_tabular_items,
            "expanded_object_count_before_cap": sum(
                len(expanded_evidence.get(k, [])) for k in self.BUCKETS
            ),
            "object_count_after_cap": sum(len(selected[k]) for k in self.BUCKETS),
            "visual_tabular_count_after_cap": (
                len(selected["figures"]) + len(selected["tables"])
            ),
            "counts_after_cap": {k: len(selected[k]) for k in self.BUCKETS},
        }
        return selected, report

    def build_context(
        self,
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        render_payload: Dict[str, Any],
    ) -> Tuple[str, Set[str], Dict[str, Any]]:
        blocks = build_llm_evidence_blocks(
            expanded_evidence=expanded_evidence,
            render_payload=render_payload,
        )

        selected_text_blocks: List[str] = []
        linked_object_blocks: List[str] = []
        included_uids: Set[str] = set()
        skipped_uids: List[str] = []

        # Figure/table objects are governed by the separate linked-object budget
        # (<=5 items) and therefore do not consume the 5,120-token textual
        # evidence budget. Their serialized captions/table payloads are still
        # included in the final model message and reported separately below.
        for uid, block in blocks:
            if block.startswith("[FIGURE]") or block.startswith("[TABLE]"):
                linked_object_blocks.append(block)
                if uid:
                    included_uids.add(uid)
                continue

            candidate_blocks = selected_text_blocks + [block]
            candidate_context = "\n\n".join(candidate_blocks).strip()
            candidate_tokens = self.count_tokens(candidate_context)
            if candidate_tokens > self.max_textual_tokens:
                skipped_uids.append(uid)
                continue
            selected_text_blocks.append(block)
            if uid:
                included_uids.add(uid)

        textual_context = "\n\n".join(selected_text_blocks).strip()
        linked_context = "\n\n".join(linked_object_blocks).strip()
        context = "\n\n".join(x for x in [textual_context, linked_context] if x).strip()

        textual_tokens = self.count_tokens(textual_context)
        linked_object_tokens = self.count_tokens(linked_context)
        total_context_tokens = self.count_tokens(context)

        report = {
            "max_textual_evidence_tokens": self.max_textual_tokens,
            "textual_evidence_tokens_used": textual_tokens,
            "linked_visual_tabular_serialized_tokens": linked_object_tokens,
            "total_serialized_evidence_tokens": total_context_tokens,
            "included_evidence_uids": sorted(included_uids),
            "skipped_evidence_uids_due_to_text_budget": skipped_uids,
        }
        return context, included_uids, report


# ============================================================
# 6. Step 4 - LLM Answer Generation (minimal output)
# ============================================================

class MinimalAnswerGenerator:
    """Explicit smoke-test fallback; not used silently in research mode."""

    def generate(
        self,
        query_analysis: QueryAnalysis,
        hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
        task_mode: str = "qa",
    ) -> LLMAnswerDraft:
        mode = normalize_task_mode(task_mode)
        render_ids = {
            "figures": [safe_str(x.get("evidence_uid")) for x in expanded_evidence.get("figures", []) if safe_str(x.get("evidence_uid"))],
            "tables": [safe_str(x.get("evidence_uid")) for x in expanded_evidence.get("tables", []) if safe_str(x.get("evidence_uid"))],
            "equations": [safe_str(x.get("evidence_uid")) for x in expanded_evidence.get("equations", []) if safe_str(x.get("evidence_uid"))],
            "references": [safe_str(x.get("evidence_uid")) for x in expanded_evidence.get("references", []) if safe_str(x.get("evidence_uid"))],
        }

        if mode == "extraction":
            task_output = {
                "composition": None,
                "processing": None,
                "experimental conditions": None,
                "property": None,
            }
            return LLMAnswerDraft(
                task_mode=mode,
                task_output=task_output,
                answer="",
                render_ids=render_ids,
                generation_debug={"fallback_used": True, "reason": "minimal_smoke_test"},
            )

        if not hits:
            answer = "The available evidence is insufficient to answer the question."
        else:
            top = hits[0]
            snippet = top.text[:220].replace("\\n", " ").strip()
            answer = (
                f"Based on the retrieved evidence from section {top.section_number} "
                f"{top.section_title}, the current smoke-test answer is grounded in: {snippet}"
            )
        return LLMAnswerDraft(
            task_mode=mode,
            task_output=answer,
            answer=answer,
            render_ids=render_ids,
            generation_debug={"fallback_used": True, "reason": "minimal_smoke_test"},
        )

class RealLLMAnswerGenerator:
    """Task-aware evidence-grounded generator with optional figure-image attachment."""

    def __init__(
        self,
        llm,
        fallback_generator: Optional["MinimalAnswerGenerator"] = None,
        enable_multimodal: bool = True,
        allow_generation_fallback: bool = False,
    ):
        self.llm = llm
        self.fallback_generator = fallback_generator or MinimalAnswerGenerator()
        self.enable_multimodal = enable_multimodal
        self.allow_generation_fallback = allow_generation_fallback

    @staticmethod
    def _programmatic_render_ids(
        candidate_render_payload: Dict[str, Any],
        allowed_evidence_uids: Set[str],
    ) -> Dict[str, List[str]]:
        """Use already-resolved evidence objects rather than asking the LLM to copy IDs."""
        out = {"figures": [], "tables": [], "equations": [], "references": []}
        for bucket in out:
            for item in candidate_render_payload.get(bucket, []):
                uid = safe_str(item.get("evidence_uid")).strip()
                if uid and (not allowed_evidence_uids or uid in allowed_evidence_uids):
                    if uid not in out[bucket]:
                        out[bucket].append(uid)
        return out

    def generate(
        self,
        query_analysis: QueryAnalysis,
        hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
        llm_context: str,
        task_mode: str = "qa",
        allowed_evidence_uids: Optional[Set[str]] = None,
        candidate_render_payload: Optional[Dict[str, Any]] = None,
    ) -> LLMAnswerDraft:
        mode = normalize_task_mode(task_mode)
        allowed = set(allowed_evidence_uids or set())
        candidate_payload = candidate_render_payload or {}
        render_ids = self._programmatic_render_ids(candidate_payload, allowed)

        prompt, template_name = build_generation_prompt(
            task_mode=mode,
            query=query_analysis.original_query,
            llm_context=llm_context,
        )

        try:
            raw_text = self._invoke(
                prompt=prompt,
                candidate_render_payload=candidate_payload,
                graph=graph,
            )
            raw_text = safe_str(raw_text).strip()
            if not raw_text:
                raise ValueError("LLM returned an empty response")

            if mode == "extraction":
                parsed = extract_json_from_llm_text(raw_text)
                if not parsed:
                    raise ValueError(
                        "Structured extraction requires a valid JSON object, but the LLM response could not be parsed."
                    )
                task_output = normalize_extraction_output(parsed)
                answer = ""
                parser_name = "strict_four_field_json"
            else:
                # The benchmark QA prompt requests a natural-language answer, not an
                # auxiliary claim/render JSON wrapper.
                task_output = raw_text
                answer = raw_text
                parser_name = "plain_text"

            return LLMAnswerDraft(
                task_mode=mode,
                task_output=task_output,
                answer=answer,
                claims=[],
                render_ids=render_ids,
                generation_debug={
                    "prompt_template_name": template_name,
                    "task_output_parser": parser_name,
                    "raw_model_output": raw_text,
                    "fallback_used": False,
                },
            )

        except Exception as exc:
            if not self.allow_generation_fallback:
                raise RuntimeError(
                    f"Generation failed for task_mode={mode!r}; fallback is disabled: {exc}"
                ) from exc
            print(f"[WARN] Generation fallback enabled: {exc}")
            draft = self.fallback_generator.generate(
                query_analysis=query_analysis,
                hits=hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
                task_mode=mode,
            )
            draft.generation_debug.update({
                "prompt_template_name": template_name,
                "generation_error": str(exc),
            })
            return draft

    def _invoke(
        self,
        prompt: str,
        candidate_render_payload: Dict[str, Any],
        graph: EvidenceGraph,
    ) -> str:
        """Invoke multimodally when linked figure images are available."""
        image_uris: List[str] = []
        if self.enable_multimodal:
            for fig in candidate_render_payload.get("figures", []):
                uri = resolve_image_uri(
                    image_path=safe_str(fig.get("image_path")),
                    image_url=safe_str(fig.get("image_url")),
                    graph=graph,
                )
                if uri and uri not in image_uris:
                    image_uris.append(uri)

        if image_uris:
            try:
                from langchain_core.messages import HumanMessage

                content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for uri in image_uris:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": uri},
                    })
                resp = self.llm.invoke([HumanMessage(content=content)])
                return getattr(resp, "content", None) or str(resp)
            except Exception as exc:
                print(f"[WARN] Multimodal invocation failed; retrying text-only: {exc}")

        resp = self.llm.invoke(prompt)
        return getattr(resp, "content", None) or str(resp)

# ============================================================
# 7. Step 5 - Program Assembler
# ============================================================


class RenderAssembler:
    """
    根据 render_ids 从 graph 中提取真正对象。
    这是“固定内容由程序输出”的核心层。
    """

    def _build_render_numbering(
        self,
        payload: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, str]]:
        numbering = {
            "figures": {},
            "tables": {},
            "equations": {},
            "references": {},
        }

        for i, item in enumerate(payload.get("figures", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["figures"][uid] = f"Fig.{i}"

        for i, item in enumerate(payload.get("tables", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["tables"][uid] = f"Table {i}"

        for i, item in enumerate(payload.get("equations", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["equations"][uid] = f"Eq.{i}"

        for i, item in enumerate(payload.get("references", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["references"][uid] = f"[{i}]"

        return numbering


    def assemble(self, render_ids: Dict[str, List[str]], graph: EvidenceGraph) -> Dict[str, List[Dict[str, Any]]]:
        payload = {
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

        for evidence_uid in render_ids.get("figures", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node and safe_str(node.get("node_type")) == "figure":
                payload["figures"].append(self._build_figure_payload(node, graph))

        for evidence_uid in render_ids.get("tables", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node and safe_str(node.get("node_type")) == "table":
                payload["tables"].append(self._build_table_payload(node, graph))

        for evidence_uid in render_ids.get("equations", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node and safe_str(node.get("node_type")) == "equation":
                payload["equations"].append(self._build_equation_payload(node, graph))

        for evidence_uid in render_ids.get("references", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node and safe_str(node.get("node_type")) == "reference":
                payload["references"].append(self._build_reference_payload(node, graph))

        payload["numbering"] = self._build_render_numbering(payload)
        return payload

    def _resolve_uid(self, evidence_uid: str, graph: EvidenceGraph) -> Optional[Dict[str, Any]]:
        return graph.resolve_evidence_uid(evidence_uid)

    def _triggered_by_uids(self, node: Dict[str, Any], graph: EvidenceGraph) -> List[str]:
        return [
            graph.make_evidence_uid(safe_str(node_id))
            for node_id in ensure_list(node.get("triggered_by_chunk_ids"))
            if safe_str(node_id)
        ]

    def _build_common_source(self, graph: EvidenceGraph) -> Dict[str, Any]:
        return {
            "doc_id": graph.doc_id,
            "source_file": graph.source_file,
            "corpus_id": graph.corpus_id,
            "paper_citation": build_paper_citation(graph),
        }

    def _build_figure_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        # 当前阶段：
        # - LLM 只使用 caption
        # - image_path / image_url 仅供前端或后端渲染使用
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "figure_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_evidence_uids": self._triggered_by_uids(node, graph),

            "caption": safe_str(node.get("content")),
            "image_path": safe_str(md.get("image_path")),
            "image_url": safe_str(md.get("image_url")),
            "source": self._build_common_source(graph),
        }

    
    def _build_table_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        table_data = ensure_dict(md.get("table_data"))
        raw = safe_str(node.get("raw_content"))
        raw_html = ""
        if raw:
            try:
                raw_obj = json.loads(raw)
                raw_html = safe_str(raw_obj.get("raw_html"))
            except json.JSONDecodeError:
                raw_html = ""

        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "table_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_evidence_uids": self._triggered_by_uids(node, graph),

            "caption": safe_str(md.get("caption")) or safe_str(node.get("content")),
            "raw_html": raw_html,

            # 当前阶段 LLM 不用 data，但可以先保留给后续渲染/分析
            "data": table_data,

            "source": self._build_common_source(graph),
        }


    def _build_equation_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "equation_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_evidence_uids": self._triggered_by_uids(node, graph),

            "equation_number": safe_str(md.get("equation_number")),
            "latex": safe_str(md.get("latex")) or safe_str(node.get("content")),
            "source": self._build_common_source(graph),
        }

    def _build_reference_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "reference_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_evidence_uids": self._triggered_by_uids(node, graph),

            "reference_number": safe_str(md.get("reference_number")),
            "content": safe_str(node.get("content")),
            "source": self._build_common_source(graph),
        }


# ============================================================
# 8. Step 6 - Final Packaging
# ============================================================
def format_source_header(source: Dict[str, Any]) -> str:
    doc_id = safe_str(source.get("doc_id"))
    citation = safe_str(source.get("paper_citation"))
    source_file = safe_str(source.get("source_file"))

    parts: List[str] = []
    if doc_id:
        parts.append(f"doc_id={doc_id}")
    if citation:
        parts.append(f"citation={citation}")
    elif source_file:
        parts.append(f"source_file={source_file}")

    return " | ".join(parts)


def build_llm_evidence_blocks(
    expanded_evidence: Dict[str, List[Dict[str, Any]]],
    render_payload: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """Serialize evidence objects into independently budgetable text blocks.

    Every block exposes the interface-level ``evidence_uid``. The LLM is never
    instructed to cite document-local node IDs.
    """
    blocks: List[Tuple[str, str]] = []

    for chunk in expanded_evidence.get("chunks", []):
        uid = safe_str(chunk.get("evidence_uid"))
        section_number = safe_str(chunk.get("section_number"))
        section_title = safe_str(chunk.get("section_title"))
        role = safe_str(chunk.get("expand_role"))
        expanded_from = safe_str(chunk.get("expanded_from"))
        expanded_from_uid = ""
        if role != "primary_hit" and expanded_from and "::" in uid:
            doc_id = uid.split("::", 1)[0]
            expanded_from_uid = f"{doc_id}::{expanded_from}"

        text = safe_str(chunk.get("content")) or safe_str(chunk.get("raw_content"))
        text = normalize_text(text)
        if not text:
            continue

        section_label = f"{section_number} {section_title}".strip()
        block = (
            "[CHUNK]\n"
            f"evidence_uid: {uid}\n"
            f"role: {role}\n"
            f"expanded_from_evidence_uid: {expanded_from_uid}\n"
            f"section: {section_label}\n"
            f"text: {text}"
        )
        blocks.append((uid, block))

    for tab in render_payload.get("tables", []):
        uid = safe_str(tab.get("evidence_uid"))
        source_header = format_source_header(ensure_dict(tab.get("source")))
        label = safe_str(tab.get("original_label")) or safe_str(tab.get("display_label"))
        caption = safe_str(tab.get("caption"))
        raw_html = safe_str(tab.get("raw_html"))

        block_parts = ["[TABLE]", f"evidence_uid: {uid}"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if caption:
            block_parts.append(f"Caption: {caption}")
        if raw_html:
            block_parts.append("HTML:")
            block_parts.append(raw_html)
        blocks.append((uid, "\n".join(block_parts)))

    for fig in render_payload.get("figures", []):
        uid = safe_str(fig.get("evidence_uid"))
        source_header = format_source_header(ensure_dict(fig.get("source")))
        label = safe_str(fig.get("original_label")) or safe_str(fig.get("display_label"))
        caption = safe_str(fig.get("caption"))

        block_parts = ["[FIGURE]", f"evidence_uid: {uid}"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if caption:
            block_parts.append(f"Caption: {caption}")
        blocks.append((uid, "\n".join(block_parts)))

    for eq in render_payload.get("equations", []):
        uid = safe_str(eq.get("evidence_uid"))
        source_header = format_source_header(ensure_dict(eq.get("source")))
        label = safe_str(eq.get("original_label")) or safe_str(eq.get("display_label"))
        latex = safe_str(eq.get("latex"))

        block_parts = ["[EQUATION]", f"evidence_uid: {uid}"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if latex:
            block_parts.append(f"LaTeX: {latex}")
        blocks.append((uid, "\n".join(block_parts)))

    for ref in render_payload.get("references", []):
        uid = safe_str(ref.get("evidence_uid"))
        source_header = format_source_header(ensure_dict(ref.get("source")))
        label = safe_str(ref.get("original_label")) or safe_str(ref.get("display_label"))
        content = safe_str(ref.get("content"))

        block_parts = ["[REFERENCE]", f"evidence_uid: {uid}"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if content:
            block_parts.append(f"Content: {content}")
        blocks.append((uid, "\n".join(block_parts)))

    return blocks


def build_llm_evidence_context(
    expanded_evidence: Dict[str, List[Dict[str, Any]]],
    render_payload: Dict[str, Any],
) -> str:
    """Backward-compatible unbudgeted context helper."""
    return "\n\n".join(
        block for _, block in build_llm_evidence_blocks(expanded_evidence, render_payload)
    ).strip()


class FinalPackager:

    def _build_source_map(
        self,
        retrieval_hits: List[RetrievalHit],
        graph: EvidenceGraph,
    ) -> List[Dict[str, Any]]:
        if not retrieval_hits:
            return []

        # 当前单文献版先只输出 1 条
        return [
            {
                "source_index": 1,
                "doc_id": graph.doc_id,
                "source_file": graph.source_file,
                "citation": build_paper_citation(graph),
            }
        ]

    def package(
        self,
        query_analysis: QueryAnalysis,
        retrieval_hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        answer_draft: LLMAnswerDraft,
        render_payload: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
        retrieval_stage_debug: Optional[Dict[str, Any]] = None,
        evidence_budget_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        primary_uids = [h.evidence_uid for h in retrieval_hits if h.evidence_uid]
        included_uids = dedupe_keep_order([
            safe_str(obj.get("evidence_uid"))
            for bucket in ("chunks", "figures", "tables", "equations", "references")
            for obj in expanded_evidence.get(bucket, [])
            if safe_str(obj.get("evidence_uid"))
        ])
        linked_render_uids = dedupe_keep_order([
            safe_str(item.get("evidence_uid"))
            for bucket in ("figures", "tables", "equations", "references")
            for item in render_payload.get(bucket, [])
            if safe_str(item.get("evidence_uid"))
        ])

        mode = normalize_task_mode(answer_draft.task_mode)
        extraction = answer_draft.task_output if mode == "extraction" else None
        answer = answer_draft.answer if mode == "qa" else None

        return {
            "task_mode": mode,
            "query": query_analysis.original_query,
            "task_output": answer_draft.task_output,
            "answer": answer,
            "extraction": extraction,
            "query_analysis": asdict(query_analysis),
            "paper_citation": build_paper_citation(graph),
            "source_map": self._build_source_map(retrieval_hits, graph),
            "claims": [self._serialize_claim(c) for c in answer_draft.claims],
            "evidence_trace": {
                "primary_retrieved_evidence_uids": primary_uids,
                "included_evidence_uids": included_uids,
                "linked_render_evidence_uids": linked_render_uids,
            },
            "render_payload": render_payload,
            "generation_debug": answer_draft.generation_debug,
            "retrieval_debug": {
                "top_hits": [self._serialize_hit(h) for h in retrieval_hits],
                "expanded_counts": {k: len(v) for k, v in expanded_evidence.items()},
                "retrieval_stage": retrieval_stage_debug or {},
                "evidence_budget": evidence_budget_report or {},
            },
        }


    def _serialize_hit(self, hit: RetrievalHit) -> Dict[str, Any]:
        return {
            "evidence_uid": hit.evidence_uid,
            "node_id": hit.node_id,
            "node_type": hit.node_type,
            "source": asdict(hit.source),
            "retrieval_distance": hit.retrieval_distance,
            "retrieval_distance_semantics": "lower_is_better",
            "rerank_score": hit.rerank_score,
            "rerank_score_semantics": "higher_is_better",
            "bucket": hit.bucket,
            "section_id": hit.section_id,
            "section_number": hit.section_number,
            "section_title": hit.section_title,
            "text": hit.text,
        }

    def _serialize_claim(self, claim: Claim) -> Dict[str, Any]:
        return {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "supports": [
                {
                    "evidence_uid": s.evidence_uid,
                    "source": asdict(s.source),
                    "node_id": s.node_id,
                    "node_type": s.node_type,
                    "display_label": s.display_label,
                    "support_role": s.support_role,
                }
                for s in claim.supports
            ],
        }


# ============================================================
# 9. 总控 Pipeline
# ============================================================
class MainPipeline:
    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        retriever: Any,
        expander: GraphExpander,
        answer_generator: Any,
        assembler: RenderAssembler,
        packager: FinalPackager,
        budget_manager: Optional[EvidenceBudgetManager] = None,
        enable_structure: bool = True,
        enable_association: bool = True,
    ):
        self.query_analyzer = query_analyzer
        self.retriever = retriever
        self.expander = expander
        self.answer_generator = answer_generator
        self.assembler = assembler
        self.packager = packager
        self.budget_manager = budget_manager
        self.enable_structure = bool(enable_structure)
        self.enable_association = bool(enable_association)

    @staticmethod
    def _filter_expanded_by_uids(
        expanded: Dict[str, List[Dict[str, Any]]],
        allowed_uids: Set[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        return {
            bucket: [
                obj for obj in expanded.get(bucket, [])
                if safe_str(obj.get("evidence_uid")) in allowed_uids
            ]
            for bucket in ("chunks", "figures", "tables", "equations", "references")
        }

    @staticmethod
    def _filter_render_payload_by_uids(
        payload: Dict[str, Any],
        allowed_uids: Set[str],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "figures": [], "tables": [], "equations": [], "references": []
        }
        for bucket in out:
            out[bucket] = [
                item for item in payload.get(bucket, [])
                if safe_str(item.get("evidence_uid")) in allowed_uids
            ]
        return out

    def run(
        self,
        query: str,
        graph: EvidenceGraph,
        task_mode: str = "qa",
        retrieval_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one evidence-grounded query.

        ``top_k`` is retained as a legacy alias for ``retrieval_top_k``. New code
        should specify ``retrieval_top_k`` and ``rerank_top_n`` separately so the
        two-stage retrieval configuration remains explicit.
        """
        if retrieval_top_k is None and top_k is not None:
            retrieval_top_k = top_k

        task_mode = normalize_task_mode(task_mode)

        # Step 1: query understanding / structural localization
        query_analysis = self.query_analyzer.analyze(query)

        if self.enable_structure:
            retrieval_buckets = list(query_analysis.retrieval_buckets)
        else:
            # w/o Structure: remove section-level localization and retrieve
            # across the complete document-level section index space.
            retrieval_buckets = [
                "intro",
                "method",
                "results",
                "discussion",
                "results_discussion",
                "conclusion",
                "other",
            ]

        # Step 2: initial FAISS retrieval -> BGE reranking
        if isinstance(self.retriever, FaissBucketedRetriever):
            retrieval_hits = self.retriever.search(
                query=query_analysis.normalized_query,
                buckets=retrieval_buckets,
                graph=graph,
                retrieval_top_k=retrieval_top_k,
                rerank_top_n=rerank_top_n,
            )
            retrieval_stage_debug = dict(self.retriever.last_debug)
        else:
            fallback_k = int(rerank_top_n or retrieval_top_k or top_k or 10)
            retrieval_hits = self.retriever.search(
                query=query_analysis.normalized_query,
                buckets=retrieval_buckets,
                graph=graph,
                top_k=fallback_k,
            )
            retrieval_stage_debug = {
                "mode": "lexical_fallback",
                "returned_count": len(retrieval_hits),
            }

        retrieval_stage_debug["structure_enabled"] = self.enable_structure
        retrieval_stage_debug["association_enabled"] = self.enable_association
        retrieval_stage_debug["query_analyzer_retrieval_buckets"] = list(
            query_analysis.retrieval_buckets
        )
        retrieval_stage_debug["actual_retrieval_buckets"] = list(retrieval_buckets)

        # Step 3: graph expansion from reranked seed evidence
        expanded_raw = self.expander.expand(
            hits=retrieval_hits,
            graph=graph,
            query_analysis=query_analysis,
            enable_association=self.enable_association,
        )

        # Step 4: evidence-object cap and linked visual/tabular cap
        if self.budget_manager is not None:
            expanded_budgeted, object_budget_report = self.budget_manager.apply_object_cap(
                expanded_evidence=expanded_raw,
                hits=retrieval_hits,
            )
        else:
            expanded_budgeted = expanded_raw
            object_budget_report = {
                "budgeting_enabled": False,
                "expanded_object_count_before_cap": sum(len(v) for v in expanded_raw.values()),
                "object_count_after_cap": sum(len(v) for v in expanded_raw.values()),
            }

        # Candidate render objects use evidence_uid at the interface boundary.
        pre_render_ids = {
            "figures": [safe_str(x.get("evidence_uid")) for x in expanded_budgeted.get("figures", [])],
            "tables": [safe_str(x.get("evidence_uid")) for x in expanded_budgeted.get("tables", [])],
            "equations": [safe_str(x.get("evidence_uid")) for x in expanded_budgeted.get("equations", [])],
            "references": [safe_str(x.get("evidence_uid")) for x in expanded_budgeted.get("references", [])],
        }
        pre_render_payload = self.assembler.assemble(
            render_ids=pre_render_ids,
            graph=graph,
        )

        # Step 5: final 5,120-token textual evidence budget.
        if self.budget_manager is not None:
            llm_context, allowed_evidence_uids, text_budget_report = self.budget_manager.build_context(
                expanded_evidence=expanded_budgeted,
                render_payload=pre_render_payload,
            )
        else:
            llm_context = build_llm_evidence_context(
                expanded_evidence=expanded_budgeted,
                render_payload=pre_render_payload,
            )
            allowed_evidence_uids = {
                safe_str(obj.get("evidence_uid"))
                for bucket in ("chunks", "figures", "tables", "equations", "references")
                for obj in expanded_budgeted.get(bucket, [])
                if safe_str(obj.get("evidence_uid"))
            }
            text_budget_report = {"budgeting_enabled": False}

        context_evidence = self._filter_expanded_by_uids(
            expanded_budgeted,
            allowed_evidence_uids,
        )
        context_render_payload = self._filter_render_payload_by_uids(
            pre_render_payload,
            allowed_evidence_uids,
        )

        evidence_budget_report = {**object_budget_report, **text_budget_report}

        # Step 6: evidence-grounded generation; linked figure images are attached
        # to the multimodal request when available.
        if isinstance(self.answer_generator, RealLLMAnswerGenerator):
            answer_draft = self.answer_generator.generate(
                query_analysis=query_analysis,
                hits=retrieval_hits,
                expanded_evidence=context_evidence,
                graph=graph,
                llm_context=llm_context,
                task_mode=task_mode,
                allowed_evidence_uids=allowed_evidence_uids,
                candidate_render_payload=context_render_payload,
            )
        else:
            answer_draft = self.answer_generator.generate(
                query_analysis=query_analysis,
                hits=retrieval_hits,
                expanded_evidence=context_evidence,
                graph=graph,
                task_mode=task_mode,
            )

        # Step 7: program-side assembly of evidence UIDs already resolved in the final evidence set.
        render_payload = self.assembler.assemble(
            render_ids=answer_draft.render_ids,
            graph=graph,
        )

        # Step 8: final package + explicit reproducibility/debug metadata.
        final_output = self.packager.package(
            query_analysis=query_analysis,
            retrieval_hits=retrieval_hits,
            expanded_evidence=context_evidence,
            answer_draft=answer_draft,
            render_payload=render_payload,
            graph=graph,
            retrieval_stage_debug=retrieval_stage_debug,
            evidence_budget_report=evidence_budget_report,
        )
        final_output["retrieval_debug"]["llm_context"] = llm_context
        return final_output


# ============================================================
# 10. 构建入口
# ============================================================
def build_pipeline_from_output_dir(
    config: PipelineConfig,
    corpus_id: Optional[str] = None,
) -> Tuple[MainPipeline, EvidenceGraph]:
    output_dir = Path(config.output_dir)
    graph_path = output_dir / config.graph_filename
    retrieval_path = output_dir / config.retrieval_docs_filename
    index_dir = output_dir / config.faiss_dirname

    if not graph_path.exists():
        raise FileNotFoundError(f"paper_graph.json not found: {graph_path}")
    if not retrieval_path.exists():
        raise FileNotFoundError(f"retrieval_docs.jsonl not found: {retrieval_path}")
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index dir not found: {index_dir}")

    graph_data = read_json(graph_path)
    retrieval_docs = read_jsonl(retrieval_path)
    graph = EvidenceGraph(
        graph_data=graph_data,
        corpus_id=corpus_id,
        base_dir=(config.asset_base_dir or output_dir),
    )

    embeddings = build_embeddings(config)

    bucket_vectorstores: Dict[str, Any] = {}
    for bucket, index_name in config.index_names.items():
        if bucket == "metadata":
            continue
        bucket_vectorstores[bucket] = load_faiss_index(
            index_dir=index_dir,
            index_name=index_name,
            embeddings=embeddings,
        )

    metadata_vectorstore = load_faiss_index(
        index_dir=index_dir,
        index_name=config.index_names["metadata"],
        embeddings=embeddings,
    )

    reranker = BgeRerank(
        model_path=config.rerank_model_path,
        top_n=config.rerank_top_n,
        device=config.device,
    )

    llm = build_llm(config)

    pipeline = MainPipeline(
        query_analyzer=LLMQueryAnalyzer(
            llm=llm,
            fallback=QueryAnalyzer(),
        ),
        retriever=FaissBucketedRetriever(
            retrieval_docs=retrieval_docs,
            bucket_vectorstores=bucket_vectorstores,
            metadata_vectorstore=metadata_vectorstore,
            reranker=reranker,
            default_top_k=config.retrieval_top_k,
            rerank_top_n=config.rerank_top_n,
        ),
        expander=GraphExpander(),
        answer_generator=RealLLMAnswerGenerator(
            llm=llm,
            fallback_generator=MinimalAnswerGenerator(),
            enable_multimodal=config.enable_multimodal,
            allow_generation_fallback=config.allow_generation_fallback,
        ),
        assembler=RenderAssembler(),
        packager=FinalPackager(),
        budget_manager=EvidenceBudgetManager(
            tokenizer_model_path=(config.context_tokenizer_path or config.embedding_model_path),
            max_evidence_objects=config.max_evidence_objects,
            max_textual_tokens=config.max_textual_evidence_tokens,
            max_visual_tabular_items=config.max_visual_tabular_items,
        ),
        enable_structure=config.enable_structure,
        enable_association=config.enable_association,
    )
    return pipeline, graph


def build_pipeline_from_files(
    graph_json_path: str | Path,
    retrieval_jsonl_path: str | Path,
    corpus_id: Optional[str] = None,
) -> Tuple[MainPipeline, EvidenceGraph]:
    """
    fallback / skeleton 入口：
    - 不加载真实 FAISS
    - 使用 BucketedRetriever 的文本匹配兜底
    """
    graph_data = read_json(graph_json_path)
    retrieval_docs = read_jsonl(retrieval_jsonl_path)
    graph = EvidenceGraph(
        graph_data=graph_data,
        corpus_id=corpus_id,
        base_dir=Path(graph_json_path).resolve().parent,
    )

    pipeline = MainPipeline(
        query_analyzer=QueryAnalyzer(),
        retriever=BucketedRetriever(retrieval_docs=retrieval_docs),
        expander=GraphExpander(),
        answer_generator=MinimalAnswerGenerator(),
        assembler=RenderAssembler(),
        packager=FinalPackager(),
        enable_structure=True,
        enable_association=True,
    )
    return pipeline, graph

# ============================================================
# 11. CLI
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Graph-aware evidence-grounded MatSci-RAG pipeline")

    parser.add_argument("--graph", default=None, help="Path to paper_graph.json (fallback mode)")
    parser.add_argument("--retrieval_docs", default=None, help="Path to retrieval_docs.jsonl (fallback mode)")

    parser.add_argument("--output_dir", default=None, help="Directory containing graph, retrieval docs, and FAISS indexes")
    parser.add_argument("--embedding_model", default="BAAI/bge-large-en-v1.5", help="Embedding model path/name")
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large", help="Reranker model path/name")
    parser.add_argument("--llm_model", default="GLM-4.1V-Thinking-Flash", help="Generation model name")
    parser.add_argument("--env_file", default="", help="Optional .env file containing API_KEY or ZHIPUAI_API_KEY")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    parser.add_argument("--retrieval_top_k", type=int, default=20, help="Initial FAISS candidate depth")
    parser.add_argument("--rerank_top_n", type=int, default=10, help="Evidence units retained after reranking")
    parser.add_argument("--max_evidence_objects", type=int, default=20)
    parser.add_argument("--max_textual_tokens", type=int, default=5120)
    parser.add_argument("--max_visual_tabular_items", type=int, default=5)
    parser.add_argument("--no_multimodal", action="store_true", help="Disable figure-image attachment to the LLM")
    parser.add_argument(
        "--no_structure",
        action="store_true",
        help="Ablation: disable structure-aware section localization",
    )
    parser.add_argument(
        "--no_association",
        action="store_true",
        help="Ablation: disable explicit graph-based evidence association",
    )
    parser.add_argument("--asset_base_dir", default="", help="Base directory for relative figure/image paths")

    parser.add_argument("--task", choices=["extraction", "qa"], default="qa", help="Benchmark task mode")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--allow_generation_fallback", action="store_true", help="Allow synthetic smoke-test output if LLM generation fails")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if args.output_dir:
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
            enable_structure=not args.no_structure,
            enable_association=not args.no_association,
            enable_multimodal=not args.no_multimodal,
            asset_base_dir=args.asset_base_dir,
            allow_generation_fallback=args.allow_generation_fallback,
        )
        pipeline, graph = build_pipeline_from_output_dir(config)
        result = pipeline.run(
            query=args.query,
            graph=graph,
            task_mode=args.task,
            retrieval_top_k=args.retrieval_top_k,
            rerank_top_n=args.rerank_top_n,
        )
    else:
        if not args.graph or not args.retrieval_docs:
            parser.error("Either --output_dir or both --graph and --retrieval_docs must be provided.")
        pipeline, graph = build_pipeline_from_files(
            graph_json_path=args.graph,
            retrieval_jsonl_path=args.retrieval_docs,
        )
        result = pipeline.run(query=args.query, graph=graph, task_mode=args.task)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
