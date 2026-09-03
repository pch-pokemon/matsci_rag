# MatSci-RAG

## 中文说明

MatSci-RAG 是一个面向材料科学文献的结构化 RAG 框架。与传统 flat RAG 不同，本项目将论文组织为由正文、图、表、公式和参考文献组成的结构化证据系统，并支持结构化信息抽取与文献问答（QA）。

### 仓库结构

```text
matsci_rag/
├── README.md
├── main_pipeline.py
├── build_demo.py
├── pdf_to_md.py
├── clean_md.py
├── md_json.py
├── json_split.py
├── metadata_processor.py
├── merge_metadata_to_graph.py
├── embedding_json.py
├── validate_pipeline.py
│
├── 1_一键运行处理——从pdf到向量分割后的向量存储.ipynb
├── 2_验证生成的证据图谱的引用关系.ipynb
├── 3_Matsci-rag实际抽取与QA测试.ipynb
│
├── ablation_component/
├── data/
├── doc2x/
└── examples/
```

### 主要流程

```text
PDF
→ Doc2X Markdown
→ Markdown 清洗
→ 结构化 JSON
→ Evidence Graph
→ Metadata enrichment
→ FAISS 向量索引
→ Structure-aware retrieval
→ BGE reranking
→ Graph evidence expansion
→ 多模态生成
→ Extraction / QA
```

推荐按照三个 Notebook 的顺序运行：

1. `1_一键运行处理——从pdf到向量分割后的向量存储.ipynb`  
   完成 PDF → Markdown → JSON → Evidence Graph → FAISS。

2. `2_验证生成的证据图谱的引用关系.ipynb`  
   检查 `node_id`、`evidence_uid`、图/表/公式/参考文献引用关系和 FAISS 一致性。

3. `3_Matsci-rag实际抽取与QA测试.ipynb`  
   运行 MatSci-RAG 的结构化抽取和 literature-grounded QA。

### 论文对应的核心配置

```text
Embedding model:        BGE-large-en-v1.5
Reranking model:        BGE-reranker-large
Initial retrieval:      top-k = 20
Reranked evidence:      top-n = 10
Max chunk length:       480 tokens
Evidence-object cap:    ≤20
Textual evidence:       ≤5120 tokens
Visual/tabular budget:  ≤5 items
Generation model:       GLM-4.1V-Thinking-Flash
Temperature:            0
```

### 快速运行

构建单篇论文示例数据库：

```bash
python build_demo.py \
  --input paper.pdf \
  --workspace examples/test_case \
  --doc2x_env .env \
  --llm_env .env \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --device cuda
```

验证生成结果：

```bash
python validate_pipeline.py \
  --output_dir examples/test_case/output
```

运行 QA：

```bash
python main_pipeline.py \
  --output_dir examples/test_case/output \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --llm_model GLM-4.1V-Thinking-Flash \
  --env_file .env \
  --task qa \
  --query "Your question"
```

结构化抽取只需将：

```text
--task qa
```

改为：

```text
--task extraction
```

### Ablation

`ablation_component/` 提供以下四种模式：

```text
full
wo_structure
wo_multimodal
wo_association
```

运行示例：

```bash
python ablation_component/run_ablation.py \
  --mode wo_structure \
  --output_dir examples/test_case/output \
  --query "Your question" \
  --task qa
```

### 数据说明

`data/` 用于保存公开的 benchmark 文献清单和相关元数据。论文 benchmark 共包含 500 篇文章：

```text
Superalloys      171
Semiconductors   164
Batteries        165
Total            500
```

出于数据发布范围考虑，完整 benchmark questions、reference answers 和 source-evidence annotations 不在公开仓库中提供。

### API Key

建议使用本地 `.env` 文件：

```text
DOC2X_API_KEY=...
ZHIPUAI_API_KEY=...
```

### 评估结果

用于计算文中汇总指标的匿名化 sample-level evaluation outputs 位于 `data/sample_level_results/`。

---

## English

MatSci-RAG is a structured retrieval-augmented generation framework for materials-science literature. Instead of treating papers as flat text, it organizes text, figures, tables, equations, and references into a structured evidence system for materials information extraction and literature-grounded QA.

### Repository structure

```text
matsci_rag/
├── README.md
├── main_pipeline.py
├── build_demo.py
├── pdf_to_md.py
├── clean_md.py
├── md_json.py
├── json_split.py
├── metadata_processor.py
├── merge_metadata_to_graph.py
├── embedding_json.py
├── validate_pipeline.py
├── 1_...ipynb
├── 2_...ipynb
├── 3_...ipynb
├── ablation_component/
├── data/
├── doc2x/
└── examples/
```

### Workflow

```text
PDF
→ Doc2X Markdown
→ Markdown normalization
→ Structured JSON
→ Evidence Graph
→ Metadata enrichment
→ FAISS indexing
→ Structure-aware retrieval
→ BGE reranking
→ Graph evidence expansion
→ Multimodal generation
→ Extraction / QA
```

The three notebooks are intended to be run in order:

1. Build the database from PDF to FAISS indexes.
2. Validate evidence-graph identifiers and citation relationships.
3. Run MatSci-RAG extraction and QA.

### Controlled configuration

```text
Embedding model:        BGE-large-en-v1.5
Reranking model:        BGE-reranker-large
Initial retrieval:      top-k = 20
Reranked evidence:      top-n = 10
Max chunk length:       480 tokens
Evidence-object cap:    ≤20
Textual evidence:       ≤5120 tokens
Visual/tabular budget:  ≤5 items
Generation model:       GLM-4.1V-Thinking-Flash
Temperature:            0
```

### Quick start

```bash
python build_demo.py \
  --input paper.pdf \
  --workspace examples/test_case \
  --doc2x_env .env \
  --llm_env .env \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --device cuda
```

Validation:

```bash
python validate_pipeline.py \
  --output_dir examples/test_case/output
```

QA:

```bash
python main_pipeline.py \
  --output_dir examples/test_case/output \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --llm_model GLM-4.1V-Thinking-Flash \
  --env_file .env \
  --task qa \
  --query "Your question"
```

Use `--task extraction` for structured extraction.

### Ablation

The `ablation_component/` directory supports:

```text
full
wo_structure
wo_multimodal
wo_association
```

Example:

```bash
python ablation_component/run_ablation.py \
  --mode wo_structure \
  --output_dir examples/test_case/output \
  --query "Your question" \
  --task qa
```

### Data

The public `data/` directory is used for benchmark bibliographic metadata.

```text
Superalloys      171
Semiconductors   164
Batteries        165
Total            500
```

The public release does not include the complete benchmark questions, reference answers, or source-evidence annotations.

### API keys

Use a local `.env` file:

```text
DOC2X_API_KEY=...
ZHIPUAI_API_KEY=...
```
### Evaluation results

Anonymized sample-level evaluation outputs underlying the reported aggregate metrics are available in `data/sample_level_results/`.