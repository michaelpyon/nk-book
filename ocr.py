"""
Layer 1: OCR Extraction
Reads page images, sends to Claude vision API, extracts Korean text.
"""

import base64
import glob
import logging
import os
import time

import anthropic
from PIL import Image

import config

logger = logging.getLogger("pipeline.ocr")


def get_image_files() -> list[str]:
    """Find all image files in photos/ sorted by filename."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(config.PHOTOS_DIR, ext)))
    # Sort by filename so page order is preserved
    files.sort(key=lambda f: os.path.basename(f).lower())
    return files


def convert_heic_to_jpeg(heic_path: str) -> str:
    """Convert a HEIC file to JPEG, return the new path."""
    jpeg_path = os.path.splitext(heic_path)[0] + ".jpg"
    if os.path.exists(jpeg_path):
        return jpeg_path
    try:
        # pillow-heif registers itself as a PIL plugin on import
        import pillow_heif
        pillow_heif.register_heif_opener()
        img = Image.open(heic_path)
        img.save(jpeg_path, "JPEG", quality=95)
        logger.info(f"Converted HEIC to JPEG: {jpeg_path}")
        return jpeg_path
    except ImportError:
        logger.error("pillow-heif not installed. Install it to handle HEIC files: pip install pillow-heif")
        raise
    except Exception as e:
        logger.error(f"Failed to convert HEIC file {heic_path}: {e}")
        raise


def encode_image(image_path: str) -> tuple[str, str]:
    """Base64 encode an image file. Returns (base64_data, media_type)."""
    ext = os.path.splitext(image_path)[1].lower()

    # Convert HEIC first
    if ext in (".heic",):
        image_path = convert_heic_to_jpeg(image_path)
        ext = ".jpg"

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def extract_page(client: anthropic.Anthropic, image_path: str, page_num: int) -> str:
    """Send a single page image to Claude for OCR. Returns extracted Korean text."""
    base64_data, media_type = encode_image(image_path)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=config.MODEL,
                max_tokens=4096,
                system=config.OCR_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Extract all Korean text from this page (page {page_num}).",
                            },
                        ],
                    }
                ],
            )
            text = response.content[0].text
            logger.info(f"Page {page_num}: extracted {len(text)} characters")
            return text

        except anthropic.RateLimitError:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Page {page_num}: rate limited, waiting {wait}s (attempt {attempt}/{config.MAX_RETRIES})")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.error(f"Page {page_num}: API error on attempt {attempt}/{config.MAX_RETRIES}: {e}")
            if attempt < config.MAX_RETRIES:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Page {page_num}: exhausted all {config.MAX_RETRIES} retries")


def get_page_number(image_path: str, index: int) -> int:
    """Extract page number from filename, or fall back to 1-based index."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # Try to extract a number from the filename
    digits = "".join(c for c in basename if c.isdigit())
    if digits:
        return int(digits)
    return index + 1


def run_ocr() -> dict:
    """
    Run OCR on all page images.
    Returns a summary dict with counts of successes and failures.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    image_files = get_image_files()
    total = len(image_files)

    if total == 0:
        logger.error("No image files found in photos/. Run extract_pages.py first if you have a PDF.")
        return {"total": 0, "success": 0, "failures": []}

    logger.info(f"Found {total} page images to process")

    successes = 0
    failures = []

    for i, image_path in enumerate(image_files):
        page_num = get_page_number(image_path, i)
        output_path = os.path.join(config.KOREAN_TEXT_DIR, f"page_{page_num:03d}.txt")

        # Resume support: skip if output already exists
        if os.path.exists(output_path):
            logger.info(f"Page {page_num}: already processed, skipping")
            successes += 1
            continue

        try:
            text = extract_page(client, image_path, page_num)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            successes += 1
            logger.info(f"Page {page_num}: saved to {output_path}")
        except Exception as e:
            logger.error(f"Page {page_num}: FAILED - {e}")
            failures.append({"page": page_num, "file": os.path.basename(image_path), "error": str(e)})

        # Rate limiting delay between calls
        if i < total - 1:
            time.sleep(config.RATE_LIMIT_DELAY)

    # Assemble full text
    _assemble_full_text()

    summary = {"total": total, "success": successes, "failures": failures}
    print(f"\nOCR complete. Processed {successes}/{total} pages.")
    if failures:
        print(f"Failures: {[f['page'] for f in failures]}")
    return summary


def _assemble_full_text():
    """Concatenate all page text files into full_korean_text.txt."""
    page_files = sorted(glob.glob(os.path.join(config.KOREAN_TEXT_DIR, "page_*.txt")))
    if not page_files:
        return

    full_text_parts = []
    for pf in page_files:
        basename = os.path.splitext(os.path.basename(pf))[0]
        # Extract page number from filename like page_001
        page_num = int(basename.replace("page_", ""))
        with open(pf, "r", encoding="utf-8") as f:
            text = f.read().strip()
        full_text_parts.append(f"--- PAGE {page_num} ---\n\n{text}")

    full_path = os.path.join(config.KOREAN_TEXT_DIR, "full_korean_text.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text_parts))

    logger.info(f"Assembled full Korean text: {full_path} ({len(page_files)} pages)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_ocr()
