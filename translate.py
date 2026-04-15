"""
Layer 2: Translation
Splits Korean text into chunks, translates each with sliding-window context,
maintains a glossary for consistency.
"""

import json
import logging
import os
import re
import time

import anthropic

import config

logger = logging.getLogger("pipeline.translate")


def load_glossary() -> dict:
    """Load the term glossary from disk."""
    if os.path.exists(config.GLOSSARY_PATH):
        with open(config.GLOSSARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_glossary(glossary: dict):
    """Save the term glossary to disk."""
    with open(config.GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def split_into_chunks(full_text: str) -> list[dict]:
    """
    Split the full Korean text into chunks of PAGES_PER_CHUNK pages each.
    Returns a list of dicts with 'chunk_num', 'pages', and 'text'.
    """
    # Split on page markers
    page_pattern = r"--- PAGE (\d+) ---"
    parts = re.split(f"({page_pattern})", full_text)

    # Reassemble into page objects
    pages = []
    i = 0
    while i < len(parts):
        if re.match(page_pattern, parts[i].strip()):
            page_num = int(re.search(r"\d+", parts[i]).group())
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            pages.append({"page_num": page_num, "text": text})
            i += 2
        else:
            # Leading text before any page marker (skip if empty)
            if parts[i].strip():
                pages.append({"page_num": 0, "text": parts[i].strip()})
            i += 1

    # Group into chunks
    chunks = []
    chunk_num = 1
    for start in range(0, len(pages), config.PAGES_PER_CHUNK):
        chunk_pages = pages[start : start + config.PAGES_PER_CHUNK]
        page_nums = [p["page_num"] for p in chunk_pages]
        combined_text = "\n\n".join(
            f"--- PAGE {p['page_num']} ---\n\n{p['text']}" for p in chunk_pages
        )
        chunks.append({
            "chunk_num": chunk_num,
            "pages": page_nums,
            "text": combined_text,
        })
        chunk_num += 1

    return chunks


def translate_chunk(
    client: anthropic.Anthropic,
    korean_text: str,
    prev_translation: str,
    glossary: dict,
    chunk_num: int,
    total_chunks: int,
) -> str:
    """Translate a single chunk of Korean text to English."""

    # Build the user message with context
    user_parts = []

    if glossary:
        glossary_str = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        user_parts.append(f"GLOSSARY OF ESTABLISHED TRANSLATIONS:\n{glossary_str}")

    if prev_translation:
        # Include last ~500 chars of previous translation for continuity
        context = prev_translation[-2000:] if len(prev_translation) > 2000 else prev_translation
        user_parts.append(f"PREVIOUS CHUNK'S ENGLISH TRANSLATION (for continuity):\n{context}")

    user_parts.append(f"KOREAN TEXT TO TRANSLATE (chunk {chunk_num}/{total_chunks}):\n{korean_text}")

    user_message = "\n\n---\n\n".join(user_parts)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=config.MODEL,
                max_tokens=8192,
                system=config.TRANSLATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            translation = response.content[0].text
            logger.info(f"Chunk {chunk_num}: translated {len(translation)} characters")
            return translation

        except anthropic.RateLimitError:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Chunk {chunk_num}: rate limited, waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.error(f"Chunk {chunk_num}: API error attempt {attempt}: {e}")
            if attempt < config.MAX_RETRIES:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Chunk {chunk_num}: exhausted retries")


def extract_glossary_terms(
    client: anthropic.Anthropic,
    korean_text: str,
    english_text: str,
    existing_glossary: dict,
) -> dict:
    """Extract new proper nouns and recurring terms from a translated chunk."""
    glossary_str = json.dumps(existing_glossary, ensure_ascii=False, indent=2) if existing_glossary else "{}"

    user_message = (
        f"EXISTING GLOSSARY:\n{glossary_str}\n\n"
        f"KOREAN TEXT:\n{korean_text}\n\n"
        f"ENGLISH TRANSLATION:\n{english_text}"
    )

    try:
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=2048,
            system=config.GLOSSARY_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text
        # Parse JSON from response (handle markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        return {t["korean"]: t["english"] for t in result.get("new_terms", [])}
    except Exception as e:
        logger.warning(f"Glossary extraction failed: {e}")
        return {}


def run_translation() -> dict:
    """
    Run the full translation pipeline.
    Returns summary dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    full_text_path = os.path.join(config.KOREAN_TEXT_DIR, "full_korean_text.txt")
    if not os.path.exists(full_text_path):
        logger.error("full_korean_text.txt not found. Run OCR first.")
        return {"error": "OCR output not found"}

    with open(full_text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_into_chunks(full_text)
    total = len(chunks)
    logger.info(f"Split into {total} chunks for translation")

    glossary = load_glossary()
    prev_translation = ""
    successes = 0
    failures = []

    for chunk in chunks:
        num = chunk["chunk_num"]
        output_path = os.path.join(config.TRANSLATIONS_DIR, f"chunk_{num:03d}.txt")

        # Resume support
        if os.path.exists(output_path):
            logger.info(f"Chunk {num}/{total}: already translated, skipping")
            with open(output_path, "r", encoding="utf-8") as f:
                prev_translation = f.read()
            successes += 1
            continue

        print(f"Translating chunk {num}/{total}... ", end="", flush=True)

        try:
            translation = translate_chunk(
                client, chunk["text"], prev_translation, glossary, num, total
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(translation)

            # Extract new glossary terms
            new_terms = extract_glossary_terms(client, chunk["text"], translation, glossary)
            if new_terms:
                glossary.update(new_terms)
                save_glossary(glossary)
                logger.info(f"Chunk {num}: added {len(new_terms)} glossary terms")

            prev_translation = translation
            successes += 1
            print("done.")

        except Exception as e:
            logger.error(f"Chunk {num}: FAILED - {e}")
            failures.append({"chunk": num, "error": str(e)})
            print(f"FAILED: {e}")

        # Rate limiting
        time.sleep(config.RATE_LIMIT_DELAY)

    # Assemble full translation
    _assemble_full_translation(total)

    summary = {"total": total, "success": successes, "failures": failures}
    print(f"\nTranslation complete. {successes}/{total} chunks translated.")
    if failures:
        print(f"Failed chunks: {[f['chunk'] for f in failures]}")
    return summary


def _assemble_full_translation(total_chunks: int):
    """Concatenate all chunk translations into full_english_translation.txt."""
    parts = []
    for i in range(1, total_chunks + 1):
        path = os.path.join(config.TRANSLATIONS_DIR, f"chunk_{i:03d}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read().strip())

    if parts:
        full_path = os.path.join(config.TRANSLATIONS_DIR, "full_english_translation.txt")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))
        logger.info(f"Assembled full English translation: {full_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_translation()
