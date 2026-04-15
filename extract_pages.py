"""
Utility: Extract page images from a PDF file into photos/ directory.
Run this before the pipeline if your source is a PDF rather than individual photos.

Usage:
    python extract_pages.py "path/to/book.pdf"
    python extract_pages.py  # auto-detects PDF in NK-Book directory
"""

import glob
import os
import sys

import config


def extract_pages(pdf_path: str, output_dir: str = config.PHOTOS_DIR, dpi: int = 300):
    """Extract each page of a PDF as a JPEG image."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("pdf2image not installed. Install it: pip install pdf2image")
        print("You also need poppler: brew install poppler (macOS)")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting pages from: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {dpi}")

    images = convert_from_path(pdf_path, dpi=dpi)
    total = len(images)
    print(f"Found {total} pages")

    for i, image in enumerate(images):
        page_num = i + 1
        output_path = os.path.join(output_dir, f"page_{page_num:03d}.jpg")

        if os.path.exists(output_path):
            print(f"  Page {page_num}/{total}: already exists, skipping")
            continue

        image.save(output_path, "JPEG", quality=95)
        print(f"  Page {page_num}/{total}: saved")

    print(f"\nExtracted {total} pages to {output_dir}")


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Auto-detect PDF in the NK-Book directory
        pdfs = glob.glob(os.path.join(config.BASE_DIR, "*.pdf"))
        if not pdfs:
            print("No PDF found. Pass the PDF path as an argument:")
            print("  python extract_pages.py /path/to/book.pdf")
            sys.exit(1)
        pdf_path = pdfs[0]
        print(f"Auto-detected PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    extract_pages(pdf_path)


if __name__ == "__main__":
    main()
