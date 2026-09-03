# MatSci-RAG 消融实验组件

本文件夹仅包含精简的评测侧组件。共享的检索（retrieval）、重排（reranking）、证据预算（evidence budgeting）、提示词（prompts）、生成模型、组装器（assembler）及打包逻辑仍保留在仓库根目录的 `main_pipeline.py` 中。

## 模式（Modes）

- `full`：完整的 MatSci-RAG。
- `wo_structure`：移除章节级定位（section-level localization），跨所有章节桶（section buckets）进行检索。
- `wo_multimodal`：移除图表证据，并禁用图片附件。
- `wo_association`：移除显式的文本–图/表图关联（graph association），同时允许图/表通过标题级语义匹配被独立检索。

## 示例（Example）

```bash
python ablation_component/run_ablation.py --mode full --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_structure --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_multimodal --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_association --output_dir examples/test_case/output --query "..." --task qa
```

所有模式均使用相同的查询集、模型设置、检索深度、重排深度、证据预算及生成设置。

# MatSci-RAG ablation components

This folder contains only thin evaluation-side components. The shared retrieval,
reranking, evidence budgeting, prompts, generation model, assembler, and packaging
logic remain in the repository-level `main_pipeline.py`.

## Modes

- `full`: complete MatSci-RAG.
- `wo_structure`: removes section-level localization and retrieves across all section buckets.
- `wo_multimodal`: removes figure/table evidence and disables image attachment.
- `wo_association`: removes explicit text–figure/table graph association while allowing
  figures/tables to be retrieved independently through caption-level semantic matching.

## Example

```bash
python ablation_component/run_ablation.py --mode full --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_structure --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_multimodal --output_dir examples/test_case/output --query "..." --task qa
python ablation_component/run_ablation.py --mode wo_association --output_dir examples/test_case/output --query "..." --task qa
```

Use the same query set, model settings, retrieval depth, reranking depth, evidence budgets, and generation settings for all modes.


