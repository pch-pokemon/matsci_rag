# -*- coding: utf-8 -*-
"""
pure_md_json.py

职责：
1. 读取已经 clean 过的 markdown
2. 提取 metadata_block / main_text
3. 提取 equations / figures / tables
4. 构建 section hierarchy
5. 切分正文与 additional information
6. 输出纯结构化 JSON

注意：
- 不再承担 clean_md 的职责
- 不依赖 LLM
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# =========================
# 0.1 图表编号统一模式
# =========================
# 支持：
# 1
# 1.2
# S1
# S1.2
# 1a
# A1
# I / II / III / IV ...
FIG_TABLE_NUM_PATTERN = r'(?:[A-Z]?\d+(?:\.\d+)*[a-zA-Z]?|[IVXLCDM]+)'

# =========================
# 0. IO
# =========================
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# =========================
# 1. 基础切分
# =========================
FRONT_MATTER_HEADINGS = {
    "abstract", "keywords", "key words", "highlights", "graphical abstract",
    "author information", "authors", "affiliations",
}

BODY_START_HINTS = (
    "introduction", "background", "materials and methods", "materials & methods",
    "methods", "methodology", "experimental", "experimental procedure",
    "experimental procedures", "experimental details", "results",
    "results and discussion", "results & discussion", "discussion",
)


def _normalized_heading_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip().lower()
    title = re.sub(r"^[0-9]+(?:\.[0-9]+)*\.?\s*", "", title)
    return title.strip()


def find_introduction_heading(md_text: str) -> Optional[re.Match]:
    return re.search(
        r'^##\s+Introduction\b',
        md_text,
        flags=re.IGNORECASE | re.MULTILINE
    )


def locate_main_text_start(md_text: str) -> Tuple[Optional[re.Match], str]:
    """
    Locate the scientific body without requiring an explicit Introduction.
    Front-matter headings such as Abstract/Keywords remain in metadata_block.
    """
    headings = list(re.finditer(r'^##\s+(.+?)\s*$', md_text, flags=re.MULTILINE))
    if not headings:
        return None, "no_level2_heading"

    normalized = [(m, _normalized_heading_title(m.group(1))) for m in headings]

    for preferred in ("introduction", "background"):
        for m, title in normalized:
            if title == preferred or title.startswith(preferred + " "):
                return m, preferred

    for m, title in normalized:
        if title in FRONT_MATTER_HEADINGS:
            continue
        if any(title == hint or title.startswith(hint + " ") for hint in BODY_START_HINTS):
            return m, "recognized_body_heading"

    # ADDITIONAL_SECTION_TITLES is defined later in this module and available at call time.
    for m, title in normalized:
        if title not in FRONT_MATTER_HEADINGS and title not in ADDITIONAL_SECTION_TITLES:
            return m, "first_non_front_matter_heading"

    return None, "no_body_heading"


def split_metadata_and_maintext(md_text: str) -> Tuple[str, str]:
    start_match, _ = locate_main_text_start(md_text)
    if start_match:
        start = start_match.start()
        return md_text[:start].strip(), md_text[start:].strip()
    return md_text.strip(), ""


# =========================
# 2. equations 提取
# =========================
def extract_equation_number(latex_body: str) -> Optional[str]:
    m = re.search(r'\\tag\{([^}]+)\}', latex_body)
    return m.group(1).strip() if m else None


def extract_equations_and_replace(md_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    提取块公式 $$...$$
    并在正文中替换成 [EQUATION:eq_x]
    """
    equations: List[Dict[str, Any]] = []
    counter = 1

    pattern = re.compile(r'\$\$(.*?)\$\$', flags=re.DOTALL)

    def repl(match: re.Match) -> str:
        nonlocal counter
        body = match.group(1).strip()
        eq_id = f"eq_{counter}"
        eq_num = extract_equation_number(body) or str(counter)

        equations.append({
            "equation_id": eq_id,
            "equation_number": eq_num,
            "equation_label": f"Equation {eq_num}",
            "latex": body
        })

        counter += 1
        return f"[EQUATION:{eq_id}]"

    new_text = pattern.sub(repl, md_text)
    return new_text, equations


# =========================
# 3. HTML table 解析
# =========================

def _safe_span(value: Any) -> int:
    try:
        v = int(str(value).strip())
        return v if v > 0 else 1
    except Exception:
        return 1


def _cell_text(cell) -> str:
    """
    保守提取单元格文本：
    - 保留单元格内部顺序
    - 压缩多余空白
    """
    return " ".join(cell.stripped_strings)


def _expand_html_table_to_grid(table) -> List[List[Dict[str, Any]]]:
    """
    将 HTML table 展开成真正的二维网格。
    每个位置都会被 rowspan / colspan 正确占位。

    返回：
    [
      [cell_obj, cell_obj, ...],
      ...
    ]

    cell_obj 示例：
    {
      "text": "A1",
      "tag": "td",
      "rowspan": 1,
      "colspan": 3,
      "is_header": False,
      "source_row": 0,
      "source_col": 1,
      "is_anchor": True
    }
    """
    trs = table.find_all("tr")
    grid: List[List[Dict[str, Any]]] = []

    # 用来存放由于 rowspan 延续到后续行的占位单元
    pending: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for r_idx, tr in enumerate(trs):
        row: List[Dict[str, Any]] = []
        c_idx = 0

        # 先填入前面行 rowspan 留下来的占位
        while (r_idx, c_idx) in pending:
            row.append(pending.pop((r_idx, c_idx)))
            c_idx += 1

        # 注意：recursive=False，避免嵌套表误入
        cells = tr.find_all(["th", "td"], recursive=False)

        for cell in cells:
            while (r_idx, c_idx) in pending:
                row.append(pending.pop((r_idx, c_idx)))
                c_idx += 1

            text = _cell_text(cell)
            rowspan = _safe_span(cell.get("rowspan", 1))
            colspan = _safe_span(cell.get("colspan", 1))
            tag = cell.name.lower()

            base = {
                "text": text,
                "tag": tag,
                "rowspan": rowspan,
                "colspan": colspan,
                "is_header": tag == "th",
            }

            # 当前行写入 colspan 展开
            for dc in range(colspan):
                cell_obj = dict(base)
                cell_obj["source_row"] = r_idx
                cell_obj["source_col"] = c_idx + dc
                cell_obj["is_anchor"] = (dc == 0)
                row.append(cell_obj)

            # 后续行写入 rowspan 占位
            for dr in range(1, rowspan):
                for dc in range(colspan):
                    occupied = dict(base)
                    occupied["source_row"] = r_idx
                    occupied["source_col"] = c_idx + dc
                    occupied["is_anchor"] = (dc == 0)
                    occupied["from_rowspan"] = True
                    pending[(r_idx + dr, c_idx + dc)] = occupied

            c_idx += colspan

        # 行尾如果还有 pending，也继续补齐
        while (r_idx, c_idx) in pending:
            row.append(pending.pop((r_idx, c_idx)))
            c_idx += 1

        grid.append(row)

    # 补齐所有行长度
    max_cols = max((len(r) for r in grid), default=0)
    for r_idx, row in enumerate(grid):
        while len(row) < max_cols:
            row.append({
                "text": "",
                "tag": "td",
                "rowspan": 1,
                "colspan": 1,
                "is_header": False,
                "source_row": r_idx,
                "source_col": len(row),
                "is_anchor": True
            })

    return grid


def _guess_header_row_count(grid: List[List[Dict[str, Any]]]) -> int:
    """
    启发式判断表头有多少行。
    目标：
    - 识别 A1/A2 这种多级表头
    - 尽量不要把第一条真实数据行误判成表头
    """
    if not grid:
        return 0

    max_scan = min(4, len(grid))
    header_count = 0

    for i in range(max_scan):
        row = grid[i]
        texts = [cell["text"].strip() for cell in row]
        non_empty = [t for t in texts if t]

        if not non_empty:
            header_count += 1
            continue

        th_ratio = sum(1 for cell in row if cell["tag"] == "th") / len(row)

        unique_non_empty = len(set(non_empty))
        repeated_exists = unique_non_empty < len(non_empty)

        numeric_like = 0
        for t in non_empty:
            if re.fullmatch(r'[-–—]|'
                            r'\d+(?:\.\d+)?'
                            r'(?:±\d+(?:\.\d+)?)?'
                            r'%?', t):
                numeric_like += 1
        numeric_ratio = numeric_like / len(non_empty) if non_empty else 0.0

        # 明确 th 主导，视为表头
        if th_ratio >= 0.5:
            header_count += 1
            continue

        # 第一行常见分组表头：重复项多、数值少
        if i == 0 and (repeated_exists or numeric_ratio < 0.5):
            header_count += 1
            continue

        # 已有表头后，下一行若仍像子表头，也继续纳入
        if header_count > 0 and repeated_exists and numeric_ratio < 0.5:
            header_count += 1
            continue

        break

    # 至少两行时，默认给 1 行表头更稳
    if header_count == 0 and len(grid) > 1:
        return 1

    return header_count


def _merge_header_rows_to_columns(header_rows: List[List[str]], total_cols: int) -> List[str]:
    """
    将多行表头合并成单行列名。
    例如：
    ["", "A1", "A1", "A1", "A2", "A2", "A2"]
    ["", "γ′", "γ",  "K",  "γ′", "γ",  "K"]
    ->
    ["col_1", "A1_γ′", "A1_γ", "A1_K", "A2_γ′", "A2_γ", "A2_K"]
    """
    columns: List[str] = []

    for c in range(total_cols):
        parts: List[str] = []
        last = None

        for hr in header_rows:
            val = hr[c].strip() if c < len(hr) else ""
            if not val:
                continue
            if val == last:
                continue
            parts.append(val)
            last = val

        columns.append("_".join(parts) if parts else f"col_{c + 1}")

    return columns


def parse_html_table_to_json(html: str) -> Dict[str, Any]:
    """
    表格解析：
    1. 真正展开 rowspan + colspan
    2. 输出完整 grid
    3. 自动识别 header_rows
    4. columns 为合并后的多级表头
    5. rows 为真实数据区

    返回格式：
    {
      "columns": [...],
      "rows": [...],
      "grid": [...],
      "header_rows": [...],
      "data_start_row": 0
    }
    """
    result = {
        "columns": [],
        "rows": [],
        "grid": [],
        "header_rows": [],
        "data_start_row": 0
    }

    if not html.strip():
        return result
    if BeautifulSoup is None:
        raise RuntimeError(
            "HTML table parsing requires beautifulsoup4. Install it with "
            "`pip install beautifulsoup4`."
        )

    try:
        # Built-in parser avoids making lxml a hard dependency.
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return result

        grid_cells = _expand_html_table_to_grid(table)
        if not grid_cells:
            return result

        text_grid: List[List[str]] = [
            [cell["text"] for cell in row]
            for row in grid_cells
        ]

        header_count = _guess_header_row_count(grid_cells)
        total_cols = max((len(r) for r in text_grid), default=0)

        header_rows = text_grid[:header_count]
        data_rows = text_grid[header_count:]

        columns = _merge_header_rows_to_columns(header_rows, total_cols)

        result["grid"] = text_grid
        result["header_rows"] = header_rows
        result["data_start_row"] = header_count
        result["columns"] = columns
        result["rows"] = data_rows

        return result

    except Exception:
        return result
    
# =========================
# 4. figures / tables 提取
# =========================
def extract_tables(md_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    clean_md 后 table 预期格式：
    # Table 1 Caption...
    <table>...</table>

    关键原则：
    - table 的边界由 <table>...</table> 决定
    - </table> 后面的正文绝不能并入 caption
    """
    tables: List[Dict[str, Any]] = []

    table_pattern = re.compile(
        r'(?P<full>'
        r'(?P<header>^#\s*Table\s*\.?\s*(?:' + FIG_TABLE_NUM_PATTERN + r').*?)\s*\n'
        r'(?P<html><table>.*?</table>)'
        r')',
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    matches = list(table_pattern.finditer(md_text))

    for idx, m in enumerate(matches, 1):
        header = (m.group("header") or "").strip()
        html = (m.group("html") or "").strip()

        number_match = re.search(
            r'Table\s*\.?\s*(' + FIG_TABLE_NUM_PATTERN + r')',
            header,
            flags=re.IGNORECASE
        )
        table_number = number_match.group(1) if number_match else str(idx)

        caption = re.sub(r'^#\s*', '', header).strip()

        tables.append({
            "table_id": f"table_{idx}",
            "table_number": f"Table {table_number}",
            "caption": caption,
            "raw_html": html,
            "data": parse_html_table_to_json(html)
        })

    cleaned = md_text
    for m in reversed(matches):
        start, end = m.span()
        cleaned = cleaned[:start] + "\n" + cleaned[end:]

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return tables, cleaned

def extract_figures(md_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    clean_md 后 figure 预期格式：
    ![...](path)
    # Fig. 1. caption
    或
    # Fig. I. caption
    或
    # Fig. S1. caption
    """
    figures: List[Dict[str, Any]] = []

    figure_pattern = re.compile(
        r'(?P<full>'
        r'(?:(?P<img>^!\[.*?\]\((?P<img_path>.*?)\)\s*\n)?)'
        r'(?P<header>^#\s*(?:Fig\.?|Figure)\s+(?:' + FIG_TABLE_NUM_PATTERN + r').*?$)'
        r')',
        flags=re.IGNORECASE | re.MULTILINE
    )

    matches = list(figure_pattern.finditer(md_text))
    for idx, m in enumerate(matches, 1):
        header = (m.group("header") or "").strip()
        img_path = (m.group("img_path") or "").strip()

        number_match = re.search(
            r'(?:Fig\.?|Figure)\s+(' + FIG_TABLE_NUM_PATTERN + r')',
            header,
            flags=re.IGNORECASE
        )
        fig_number = number_match.group(1) if number_match else str(idx)
        caption = re.sub(r'^#\s*', '', header).strip()

        figures.append({
            "figure_id": f"fig_{idx}",
            "figure_number": f"Fig. {fig_number}",
            "caption": caption,
            "image_path": img_path,
            "image_url": img_path
        })

    cleaned = md_text
    for m in reversed(matches):
        start, end = m.span()
        cleaned = cleaned[:start] + "\n" + cleaned[end:]

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return figures, cleaned

def extract_figures_tables(md_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    tables, text_wo_tables = extract_tables(md_text)
    figures, text_wo_figures_tables = extract_figures(text_wo_tables)
    return figures, tables, text_wo_figures_tables


# =========================
# 5. additional information 切分
# =========================
ADDITIONAL_SECTION_TITLES = {
    "declaration of competing interest",
    "declaration of competing interests",
    "competing interest",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "data availability",
    "availability of data and materials",
    "author contributions",
    "credit authorship contribution statement",
    "contribution statement",
    "supplementary materials",
    "supplementary material",
    "appendix",
    "references",
}


def split_maintext_and_additional(md_text: str) -> Tuple[str, str]:
    headings = list(re.finditer(r'^##\s+(.*)$', md_text, flags=re.MULTILINE))
    if not headings:
        return md_text.strip(), ""

    cutoff_index = None
    for m in headings:
        title = m.group(1).strip().lower()
        if title in ADDITIONAL_SECTION_TITLES:
            cutoff_index = m.start()
            break

    if cutoff_index is None:
        return md_text.strip(), ""

    return md_text[:cutoff_index].strip(), md_text[cutoff_index:].strip()


def extract_additional_information(md_text: str) -> List[Dict[str, str]]:
    if not md_text.strip():
        return []

    blocks = re.split(r'\n(?=##\s+)', md_text.strip())
    result = []

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        title = re.sub(r'^##\s*', '', lines[0]).strip()
        content = "\n".join(lines[1:]).strip()

        result.append({
            "additional_info_title": title,
            "content": content
        })

    return result


# =========================
# 6. section hierarchy
# =========================
def slugify_section_number(section_number: str) -> str:
    return section_number.replace(".", "_") if section_number else "root"


def extract_sections_with_hierarchy(md_text: str) -> List[Dict[str, Any]]:
    """
    从 ## ~ ###### 构建层级结构
    """
    lines = md_text.splitlines()
    sections: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    content_buffer: List[str] = []
    section_counters = [0] * 7  # index 0 unused

    def flush_content() -> None:
        nonlocal content_buffer
        content = "\n".join(content_buffer).strip()
        if content and stack:
            stack[-1]["content"] = content
            stack[-1]["equation_refs"] = re.findall(r'\[EQUATION:(eq_\d+)\]', content)
        content_buffer = []

    def build_hierarchy(current_section: Dict[str, Any]) -> List[Dict[str, Any]]:
        chain = []
        for sec in stack:
            chain.append({
                "section_id": sec["section_id"],
                "section_number": sec["section_number"],
                "section_title": sec["section_title"],
                "section_title_full": sec["section_title_full"]
            })
        chain.append({
            "section_id": current_section["section_id"],
            "section_number": current_section["section_number"],
            "section_title": current_section["section_title"],
            "section_title_full": current_section["section_title_full"]
        })
        return chain

    def insert_section(hash_count: int, title: str) -> None:
        nonlocal stack, sections

        logical_level = hash_count - 1  # ## -> 1, ### -> 2

        section_counters[logical_level] += 1
        for i in range(logical_level + 1, len(section_counters)):
            section_counters[i] = 0

        number_parts = [
            str(section_counters[i])
            for i in range(1, logical_level + 1)
            if section_counters[i] > 0
        ]
        section_number = ".".join(number_parts)
        section_id = f"sec_{slugify_section_number(section_number)}"

        stack = stack[:logical_level - 1] if logical_level > 1 else []

        new_section = {
            "section_id": section_id,
            "section_number": section_number,
            "section_title": title.strip(),
            "section_title_full": f"{section_number} {title.strip()}".strip(),
            "level": logical_level,
            "parent_section_id": stack[-1]["section_id"] if stack else None,
            "hierarchy": [],
            "content": "",
            "equation_refs": [],
            "subsections": []
        }

        new_section["hierarchy"] = build_hierarchy(new_section)

        if logical_level == 1:
            sections.append(new_section)
        else:
            if stack:
                stack[-1]["subsections"].append(new_section)
            else:
                sections.append(new_section)

        stack.append(new_section)

    heading_pattern = re.compile(r'^(#{2,6})\s*(.*)$')

    for line in lines:
        m = heading_pattern.match(line)
        if m:
            flush_content()
            hashes, title = m.groups()
            insert_section(len(hashes), title)
        else:
            content_buffer.append(line)

    flush_content()
    return sections


# =========================
# 7. equation 上下文回填
# =========================
def attach_equation_context_to_sections(
    sections: List[Dict[str, Any]],
    equations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    eq_index = {eq["equation_id"]: eq for eq in equations}

    def walk(sec_list: List[Dict[str, Any]]) -> None:
        for sec in sec_list:
            for eq_id in sec.get("equation_refs", []):
                if eq_id in eq_index:
                    eq = eq_index[eq_id]
                    eq["section_id"] = sec["section_id"]
                    eq["section_number"] = sec["section_number"]
                    eq["section_title"] = sec["section_title"]
                    eq["section_title_full"] = sec["section_title_full"]
                    eq["hierarchy"] = sec["hierarchy"]

            if sec.get("subsections"):
                walk(sec["subsections"])

    walk(sections)
    return equations


# =========================
# 8. 主流程
# =========================
def build_json_from_markdown(
    md_text: str,
    doc_id: str = "",
    source_file: str = "",
    source_markdown: str = "",
) -> Dict[str, Any]:
    md_text = normalize_newlines(md_text)

    start_match, start_strategy = locate_main_text_start(md_text)
    if start_match:
        start = start_match.start()
        metadata_block = md_text[:start].strip()
        main_text = md_text[start:].strip()
    else:
        metadata_block = md_text.strip()
        main_text = ""

    main_text, equations = extract_equations_and_replace(main_text)
    figures, tables, main_text_clean = extract_figures_tables(main_text)

    body_text, additional_text = split_maintext_and_additional(main_text_clean)
    sections = extract_sections_with_hierarchy(body_text)
    equations = attach_equation_context_to_sections(sections, equations)
    additional_information = extract_additional_information(additional_text)

    return {
        "schema_version": "2.1-structured-markdown",
        "doc_id": doc_id,
        "source_file": source_file,
        "source_markdown": source_markdown,
        "processing_info": {
            "main_text_start_strategy": start_strategy,
            "section_count_top_level": len(sections),
            "equation_count": len(equations),
            "table_count": len(tables),
            "figure_count": len(figures),
            "additional_info_count": len(additional_information),
        },
        "metadata_block": metadata_block,
        "sections": sections,
        "equations": equations,
        "tables": tables,
        "figures": figures,
        "additional_information": additional_information,
    }


# =========================
# 9. File processing / CLI
# =========================

def _strip_processing_suffixes(stem: str) -> str:
    for suffix in (".cleaned", "_cleaned", "-cleaned"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def normalize_source_file_label(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def infer_doc_id(input_path: str | Path, source_file: str = "") -> str:
    if source_file:
        source_stem = Path(normalize_source_file_label(source_file)).stem
        if source_stem:
            return source_stem
    stem = _strip_processing_suffixes(Path(input_path).stem)
    return stem or "document"


def default_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    stem = _strip_processing_suffixes(input_path.stem)
    return input_path.with_name(f"{stem}.json")


def convert_markdown_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    doc_id: str | None = None,
    source_file: str | None = None,
) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input cleaned Markdown not found: {input_path}")

    output_path = Path(output_path) if output_path else default_output_path(input_path)
    source_label = normalize_source_file_label(source_file or input_path.name)
    resolved_doc_id = (doc_id or "").strip() or infer_doc_id(input_path, source_label)

    final_json = build_json_from_markdown(
        md_text=read_text(str(input_path)),
        doc_id=resolved_doc_id,
        source_file=source_label,
        source_markdown=input_path.name,
    )
    write_json(output_path, final_json)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cleaned scientific Markdown into the structured JSON schema "
            "consumed by the MatSci-RAG evidence-graph builder."
        )
    )
    parser.add_argument("--input", required=True, help="Cleaned Markdown path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--doc_id", default=None, help="Optional stable document ID")
    parser.add_argument(
        "--source_file",
        default=None,
        help=(
            "Original source-file label (recommended: PDF filename, e.g. test.pdf). "
            "Only the basename is stored; defaults to the cleaned Markdown filename."
        ),
    )
    args = parser.parse_args()

    output_path = convert_markdown_file(
        input_path=args.input,
        output_path=args.output,
        doc_id=args.doc_id,
        source_file=args.source_file,
    )
    print(f"[OK] structured JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
