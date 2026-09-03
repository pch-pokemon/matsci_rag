# -*- coding: utf-8 -*-
from __future__ import annotations

"""
pdf_to_md.py
============

Thin, reproducible wrapper around Doc2X (via ``pdfdeal``) for converting a PDF
into the Markdown layout expected by the MatSci-RAG preprocessing pipeline.

Responsibilities
----------------
1. Convert one PDF with Doc2X.
2. Request a ZIP whose name follows the source PDF stem.
3. Safely extract the ZIP into a per-document directory.
4. Rename Doc2X's ``output.md`` to ``<pdf_stem>.md``.
5. Preserve the extracted ``images/`` directory and other Doc2X assets.
6. Optionally remove the intermediate ZIP.
7. Reuse an existing normalized Markdown output unless ``--force`` is supplied.

Example
-------
python pdf_to_md.py \
    --input examples/test_case/test.pdf \
    --output_dir examples/test_case/doc2x_output \
    --env_file .env

Expected output
---------------
examples/test_case/doc2x_output/
└── test/
    ├── test.md
    └── images/

Credentials
-----------
The API key is resolved in this order:
- DOC2X_API_KEY
- API_KEY

An optional ``--env_file`` may be supplied. No machine-specific path is embedded
in this script.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


DEFAULT_OUTPUT_DIR = "./Output"
DEFAULT_OUTPUT_FORMAT = "md_dollar"
DEFAULT_API_KEY_ENV_CANDIDATES: tuple[str, ...] = (
    "DOC2X_API_KEY",
    "API_KEY",
)


# ============================================================
# 0. Dependency / credential helpers
# ============================================================

def _load_env_file(env_file_path: Optional[str | Path]) -> None:
    """Load an optional env file without overriding already-defined variables."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "python-dotenv is required for --env_file support. "
            "Install it with `pip install python-dotenv`."
        ) from exc

    if env_file_path:
        env_path = Path(env_file_path)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")
        load_dotenv(env_path, override=False)
    else:
        # Also allow a conventional .env in the current/project directory.
        load_dotenv(override=False)


def resolve_api_key(
    env_file_path: Optional[str | Path] = None,
    env_candidates: Sequence[str] = DEFAULT_API_KEY_ENV_CANDIDATES,
) -> str:
    _load_env_file(env_file_path)

    for name in env_candidates:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()

    names = ", ".join(env_candidates)
    raise ValueError(
        "Doc2X API key not found. Set one of the following environment "
        f"variables: {names}, or provide --env_file containing one of them."
    )


def build_doc2x_client(
    env_file_path: Optional[str | Path] = None,
    debug: bool = False,
):
    """Initialize the Doc2X client lazily so --help works without pdfdeal installed."""
    try:
        from pdfdeal import Doc2X
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "pdfdeal is required for PDF-to-Markdown conversion. "
            "Install the Doc2X/pdfdeal dependency before running this stage."
        ) from exc

    api_key = resolve_api_key(env_file_path=env_file_path)
    return Doc2X(apikey=api_key, debug=debug)


# ============================================================
# 1. ZIP handling
# ============================================================

def _is_within_directory(base_dir: Path, target: Path) -> bool:
    """Compatibility-safe path containment test used to prevent ZIP path traversal."""
    base = base_dir.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> None:
    zip_path = Path(zip_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = destination / member.filename
            if not _is_within_directory(destination, target):
                raise ValueError(
                    f"Unsafe path detected in ZIP archive: {member.filename!r}"
                )
        zf.extractall(destination)


def _find_markdown_after_extraction(paper_dir: Path) -> Path:
    """Prefer output.md, otherwise accept a single Markdown file."""
    direct = paper_dir / "output.md"
    if direct.exists():
        return direct

    output_named = sorted(
        p for p in paper_dir.rglob("output.md") if p.is_file()
    )
    if output_named:
        if len(output_named) > 1:
            print(
                f"[WARN] Multiple output.md files found; using: {output_named[0]}"
            )
        return output_named[0]

    md_files = sorted(p for p in paper_dir.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(
            f"No Markdown file found after extracting Doc2X output into: {paper_dir}"
        )
    if len(md_files) > 1:
        print(f"[WARN] Multiple Markdown files found; using: {md_files[0]}")
    return md_files[0]


def extract_and_normalize(
    zip_path: str | Path,
    pdf_path: str | Path,
    output_dir: str | Path,
    remove_zip: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Extract a Doc2X ZIP and normalize its main Markdown filename.

    Example
    -------
    test.zip -> Output/test/test.md + Output/test/images/
    """
    zip_path = Path(zip_path)
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    stem = pdf_path.stem
    paper_dir = output_dir / stem

    if overwrite and paper_dir.exists():
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    safe_extract_zip(zip_path, paper_dir)
    source_md = _find_markdown_after_extraction(paper_dir)
    final_md = paper_dir / f"{stem}.md"

    if source_md.resolve() != final_md.resolve():
        if final_md.exists():
            final_md.unlink()
        shutil.move(str(source_md), str(final_md))

    if remove_zip and zip_path.exists():
        zip_path.unlink()

    return final_md


# ============================================================
# 2. Doc2X result resolution
# ============================================================

def _iter_pathlike_values(obj: Any) -> Iterable[Path]:
    """Yield path-looking values from common Doc2X return structures."""
    if obj is None:
        return
    if isinstance(obj, (str, os.PathLike)):
        yield Path(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_pathlike_values(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_pathlike_values(value)


def locate_converted_zip(
    output_dir: str | Path,
    expected_zip_name: str,
    success_result: Any = None,
) -> Path:
    output_dir = Path(output_dir)

    expected = output_dir / expected_zip_name
    if expected.exists():
        return expected

    for candidate in _iter_pathlike_values(success_result):
        if candidate.exists() and candidate.suffix.lower() == ".zip":
            return candidate
        if not candidate.is_absolute():
            joined = output_dir / candidate
            if joined.exists() and joined.suffix.lower() == ".zip":
                return joined

    same_stem = sorted(output_dir.glob(expected_zip_name))
    if same_stem:
        return same_stem[0]

    zip_files = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(zip_files) == 1:
        print(f"[WARN] Expected ZIP name not found; using the only ZIP present: {zip_files[0]}")
        return zip_files[0]

    raise FileNotFoundError(
        f"Converted ZIP file not found. Expected: {expected}"
    )


# ============================================================
# 3. Public conversion API
# ============================================================

def pdf_to_md(
    pdf_file: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    env_file_path: Optional[str | Path] = None,
    debug: bool = False,
    remove_zip: bool = True,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    force: bool = False,
) -> Path:
    """Convert one PDF to normalized Doc2X Markdown and return the Markdown path."""
    pdf_file = Path(pdf_file)
    output_dir = Path(output_dir)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file: {pdf_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_file.stem
    final_md = output_dir / stem / f"{stem}.md"

    if final_md.exists() and not force:
        print(f"[CACHE] Reusing existing Doc2X Markdown: {final_md}")
        return final_md

    zip_name = f"{stem}.zip"

    print("=" * 80)
    print("PDF -> Markdown (Doc2X)")
    print("=" * 80)
    print(f"[PDF]          {pdf_file}")
    print(f"[OUTPUT_DIR]   {output_dir}")
    print(f"[FORMAT]       {output_format}")
    print(f"[REMOVE_ZIP]   {remove_zip}")
    print()

    client = build_doc2x_client(
        env_file_path=env_file_path,
        debug=debug,
    )

    success, failed, flag = client.pdf2file(
        pdf_file=str(pdf_file),
        output_path=str(output_dir),
        output_names=[zip_name],
        output_format=output_format,
    )

    print("\n===== Doc2X result =====")
    print("success:", success)
    print("failed :", failed)
    print("flag   :", flag)

    # Preserve the semantics used by the existing working Doc2X wrapper.
    if flag:
        raise RuntimeError(f"Doc2X conversion failed:\n{failed}")

    zip_path = locate_converted_zip(
        output_dir=output_dir,
        expected_zip_name=zip_name,
        success_result=success,
    )

    final_md = extract_and_normalize(
        zip_path=zip_path,
        pdf_path=pdf_file,
        output_dir=output_dir,
        remove_zip=remove_zip,
        overwrite=True,
    )

    print("\n===== Completed =====")
    print(f"[MD]     {final_md}")

    images_dir = final_md.parent / "images"
    if images_dir.exists():
        print(f"[IMAGES] {images_dir}")
    else:
        print("[IMAGES] No images directory found.")

    return final_md


# ============================================================
# 4. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a scientific PDF to Doc2X Markdown and normalize the extracted "
            "archive into a per-document directory."
        )
    )
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Doc2X output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--env_file",
        default=None,
        help="Optional env file containing DOC2X_API_KEY or API_KEY",
    )
    parser.add_argument(
        "--output_format",
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Doc2X output format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Doc2X debug output")
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep the intermediate Doc2X ZIP instead of deleting it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run Doc2X again even if normalized Markdown already exists",
    )
    args = parser.parse_args()

    final_md = pdf_to_md(
        pdf_file=args.input,
        output_dir=args.output_dir,
        env_file_path=args.env_file,
        debug=args.debug,
        remove_zip=not args.keep_zip,
        output_format=args.output_format,
        force=args.force,
    )
    print(f"[OK] Final Markdown: {final_md}")


if __name__ == "__main__":
    main()
