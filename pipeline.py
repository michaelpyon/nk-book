"""
Main orchestration script for the NK-Book translation pipeline.

Usage:
    python pipeline.py --step all        # Run full pipeline
    python pipeline.py --step ocr        # Run only OCR
    python pipeline.py --step translate   # Run only translation
    python pipeline.py --step validate    # Run only validation
    python pipeline.py --step assemble    # Assemble final output
"""

import argparse
import glob
import json
import logging
import os
import sys
import re

import config
from ocr import run_ocr
from translate import run_translation, split_into_chunks
from validate import run_validation


def setup_logging():
    """Configure logging to both file and console."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # File handler: everything
    fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    # Console handler: INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Also configure submodule loggers to propagate
    for name in ("pipeline.ocr", "pipeline.translate", "pipeline.validate"):
        sub = logging.getLogger(name)
        sub.setLevel(logging.DEBUG)
        sub.addHandler(fh)
        sub.addHandler(ch)


def load_progress() -> dict:
    """Load pipeline progress state."""
    if os.path.exists(config.PROGRESS_PATH):
        with open(config.PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ocr": None, "translate": None, "validate": None, "assemble": None}


def save_progress(progress: dict):
    """Save pipeline progress state."""
    with open(config.PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def step_ocr(progress: dict) -> dict:
    """Run OCR step."""
    print("\n=== LAYER 1: OCR EXTRACTION ===\n")
    result = run_ocr()
    progress["ocr"] = result
    save_progress(progress)
    return result


def step_translate(progress: dict) -> dict:
    """Run translation step."""
    print("\n=== LAYER 2: TRANSLATION ===\n")
    result = run_translation()
    progress["translate"] = result
    save_progress(progress)
    return result


def step_validate(progress: dict) -> dict:
    """Run validation step."""
    print("\n=== LAYER 3: VALIDATION ===\n")
    result = run_validation()
    progress["validate"] = result
    save_progress(progress)
    return result


def step_assemble(progress: dict) -> dict:
    """Assemble final output documents."""
    print("\n=== ASSEMBLY ===\n")

    # 1. Assemble final translation
    chunk_files = sorted(glob.glob(os.path.join(config.TRANSLATIONS_DIR, "chunk_*.txt")))
    if not chunk_files:
        print("No translated chunks found. Run translation first.")
        return {"error": "No translations found"}

    # Read all translations
    translations = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            translations.append(f.read().strip())

    full_translation = "\n\n".join(translations)
    final_translation_path = os.path.join(config.FINAL_DIR, "full_translation.txt")
    with open(final_translation_path, "w", encoding="utf-8") as f:
        f.write(full_translation)
    print(f"Final translation: {final_translation_path}")

    # 2. Generate QA report
    validation_files = sorted(glob.glob(os.path.join(config.VALIDATION_DIR, "chunk_*_validation.json")))
    qa_data = {
        "total_chunks": len(chunk_files),
        "validated_chunks": len(validation_files),
        "chunk_grades": {},
        "total_issues_found": 0,
        "high_severity_issues": 0,
        "chunks_needing_human_review": [],
        "total_retranslation_attempts": 0,
        "glossary": {},
    }

    for vf in validation_files:
        with open(vf, "r", encoding="utf-8") as f:
            v = json.load(f)
        chunk_num = v.get("chunk_num", "?")
        qa_data["chunk_grades"][str(chunk_num)] = v.get("quality_grade", "?")

        bt_issues = v.get("back_translation_issues", [])
        qa_data["total_issues_found"] += len(bt_issues)
        qa_data["high_severity_issues"] += sum(1 for i in bt_issues if i.get("severity") == "HIGH")

        if v.get("needs_human_review"):
            qa_data["chunks_needing_human_review"].append(chunk_num)

        qa_data["total_retranslation_attempts"] += v.get("retranslation_attempts", 0)

    # Add glossary
    if os.path.exists(config.GLOSSARY_PATH):
        with open(config.GLOSSARY_PATH, "r", encoding="utf-8") as f:
            qa_data["glossary"] = json.load(f)

    qa_path = os.path.join(config.FINAL_DIR, "qa_report.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"QA report: {qa_path}")

    # 3. Generate side-by-side document
    full_korean_path = os.path.join(config.KOREAN_TEXT_DIR, "full_korean_text.txt")
    if os.path.exists(full_korean_path):
        with open(full_korean_path, "r", encoding="utf-8") as f:
            full_korean = f.read()

        chunks = split_into_chunks(full_korean)
        side_by_side_parts = []

        for chunk in chunks:
            num = chunk["chunk_num"]
            chunk_trans_path = os.path.join(config.TRANSLATIONS_DIR, f"chunk_{num:03d}.txt")
            if os.path.exists(chunk_trans_path):
                with open(chunk_trans_path, "r", encoding="utf-8") as f:
                    english = f.read().strip()
            else:
                english = "[TRANSLATION NOT AVAILABLE]"

            # Split Korean and English into paragraphs and interleave
            korean_paragraphs = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()]
            english_paragraphs = [p.strip() for p in english.split("\n\n") if p.strip()]

            side_by_side_parts.append(f"{'=' * 60}")
            side_by_side_parts.append(f"CHUNK {num}")
            side_by_side_parts.append(f"{'=' * 60}")

            # Pair paragraphs (best effort, since counts might not match exactly)
            max_paras = max(len(korean_paragraphs), len(english_paragraphs))
            for i in range(max_paras):
                kr = korean_paragraphs[i] if i < len(korean_paragraphs) else ""
                en = english_paragraphs[i] if i < len(english_paragraphs) else ""

                if kr:
                    side_by_side_parts.append(f"[KR] {kr}")
                if en:
                    side_by_side_parts.append(f"[EN] {en}")
                side_by_side_parts.append("")

        sbs_path = os.path.join(config.FINAL_DIR, "side_by_side.txt")
        with open(sbs_path, "w", encoding="utf-8") as f:
            f.write("\n".join(side_by_side_parts))
        print(f"Side-by-side: {sbs_path}")

    result = {"assembled": True, "qa_report": qa_path}
    progress["assemble"] = result
    save_progress(progress)

    # Print summary
    print(f"\n--- PIPELINE SUMMARY ---")
    print(f"Total chunks: {qa_data['total_chunks']}")
    print(f"Validated: {qa_data['validated_chunks']}")
    print(f"Issues found: {qa_data['total_issues_found']} ({qa_data['high_severity_issues']} high severity)")
    print(f"Re-translations: {qa_data['total_retranslation_attempts']}")
    if qa_data["chunks_needing_human_review"]:
        print(f"Human review needed: chunks {qa_data['chunks_needing_human_review']}")
    print(f"Grades: {qa_data['chunk_grades']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="NK-Book Translation Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "ocr", "translate", "validate", "assemble"],
        default="all",
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    # Validate API key
    if not config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    # Ensure output directories exist
    for d in (config.KOREAN_TEXT_DIR, config.TRANSLATIONS_DIR, config.VALIDATION_DIR, config.FINAL_DIR):
        os.makedirs(d, exist_ok=True)

    setup_logging()
    progress = load_progress()

    if args.step == "all":
        step_ocr(progress)
        step_translate(progress)
        step_validate(progress)
        step_assemble(progress)
    elif args.step == "ocr":
        step_ocr(progress)
    elif args.step == "translate":
        step_translate(progress)
    elif args.step == "validate":
        step_validate(progress)
    elif args.step == "assemble":
        step_assemble(progress)

    print("\nDone.")


if __name__ == "__main__":
    main()
