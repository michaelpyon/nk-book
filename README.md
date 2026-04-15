# NK-Book: Korean Book OCR, Translation & Validation Pipeline

Extracts Korean text from scanned page photos, translates to English with tone preservation, and validates every section using three independent sub-agents.

## Setup

```bash
pip install -r requirements.txt
brew install poppler  # needed for pdf2image on macOS
export ANTHROPIC_API_KEY="your-key-here"
```

## If your source is a PDF

```bash
python extract_pages.py  # auto-detects PDF in the directory
# or
python extract_pages.py "/path/to/book.pdf"
```

This extracts each page as a 300 DPI JPEG into `photos/`.

## Running the Pipeline

```bash
# Full pipeline (OCR -> Translate -> Validate -> Assemble)
python pipeline.py --step all

# Individual steps
python pipeline.py --step ocr
python pipeline.py --step translate
python pipeline.py --step validate
python pipeline.py --step assemble
```

Every step is resumable. If it crashes mid-run, re-running picks up where it left off.

## Output

- `output/korean_text/` - Raw Korean OCR per page + concatenated full text
- `output/translations/` - English translations per chunk + full translation
- `output/validation/` - QA reports per chunk (back-translation, tone scores, editorial)
- `output/final/full_translation.txt` - The final English translation
- `output/final/qa_report.json` - Quality report with grades and issue counts
- `output/final/side_by_side.txt` - Korean and English paragraphs interleaved
- `output/glossary.json` - Consistent term mappings maintained across the book

## Pipeline Layers

**Layer 1 (OCR):** Sends each page photo to Claude's vision API, extracts Korean text exactly as written.

**Layer 2 (Translation):** Splits text into 3-page chunks, translates with sliding-window context for continuity. Maintains a glossary of proper nouns and recurring terms.

**Layer 3 (Validation):** Three independent agents check each chunk:
1. Back-translation check (English -> Korean -> compare with original for semantic drift)
2. Tone & register scoring (formality, emotion, voice, rhythm, cultural fidelity)
3. Editorial reconciliation (proposes revisions, assigns quality grade)

Chunks scoring below 3.5 or with HIGH severity issues are automatically re-translated (max 2 attempts), then flagged for human review if still below threshold.
