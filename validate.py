"""
Layer 3: Validation
Three independent sub-agents validate each translation chunk:
  Agent 1: Back-translation semantic drift check
  Agent 2: Tone & register scoring
  Agent 3: Editorial reconciliation with revisions
"""

import json
import logging
import os
import re
import time

import anthropic

import config

logger = logging.getLogger("pipeline.validate")


def _parse_json_response(raw: str) -> dict | list:
    """Parse JSON from a Claude response, handling markdown code blocks."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _api_call(client: anthropic.Anthropic, system: str, user_msg: str, max_tokens: int = 4096) -> str:
    """Make an API call with retry logic."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=config.MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = config.RETRY_DELAY * (2 ** (attempt - 1))
            logger.error(f"API error attempt {attempt}: {e}")
            if attempt < config.MAX_RETRIES:
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def agent1_back_translation(
    client: anthropic.Anthropic,
    original_korean: str,
    english_translation: str,
) -> list:
    """
    Agent 1: Back-translate English to Korean, then compare with original.
    Returns list of semantic drift issues.
    """
    # Step 1: Back-translate English -> Korean
    back_translate_msg = (
        f"ORIGINAL KOREAN (for style reference):\n{original_korean}\n\n"
        f"ENGLISH TEXT TO TRANSLATE BACK TO KOREAN:\n{english_translation}"
    )
    back_korean = _api_call(client, config.BACK_TRANSLATION_SYSTEM_PROMPT, back_translate_msg)
    time.sleep(config.RATE_LIMIT_DELAY)

    # Step 2: Compare original vs back-translated Korean
    compare_msg = (
        f"ORIGINAL Korean text:\n{original_korean}\n\n"
        f"BACK-TRANSLATED Korean text:\n{back_korean}"
    )
    raw = _api_call(client, config.COMPARISON_SYSTEM_PROMPT, compare_msg)

    try:
        issues = _parse_json_response(raw)
        if not isinstance(issues, list):
            issues = []
    except json.JSONDecodeError:
        logger.warning("Agent 1: failed to parse comparison JSON, treating as no issues")
        issues = []

    return issues


def agent2_tone_check(
    client: anthropic.Anthropic,
    original_korean: str,
    english_translation: str,
) -> dict:
    """
    Agent 2: Evaluate tone, register, voice, rhythm, and cultural fidelity.
    Returns scoring dict.
    """
    user_msg = (
        f"ORIGINAL KOREAN TEXT:\n{original_korean}\n\n"
        f"ENGLISH TRANSLATION:\n{english_translation}"
    )
    raw = _api_call(client, config.TONE_CHECK_SYSTEM_PROMPT, user_msg)

    try:
        result = _parse_json_response(raw)
    except json.JSONDecodeError:
        logger.warning("Agent 2: failed to parse tone check JSON, using default scores")
        result = {
            "scores": {
                "formality_register": {"score": 3, "notes": "parse error"},
                "emotional_tone": {"score": 3, "notes": "parse error"},
                "voice_authenticity": {"score": 3, "notes": "parse error"},
                "rhythm_pacing": {"score": 3, "notes": "parse error"},
                "cultural_fidelity": {"score": 3, "notes": "parse error"},
            },
            "overall_score": 3.0,
            "flagged_passages": [],
        }

    # Calculate overall score if not present
    if "overall_score" not in result and "scores" in result:
        scores = [v.get("score", 3) for v in result["scores"].values() if isinstance(v, dict)]
        result["overall_score"] = sum(scores) / len(scores) if scores else 3.0

    return result


def agent3_editorial(
    client: anthropic.Anthropic,
    original_korean: str,
    english_translation: str,
    back_translation_issues: list,
    tone_scores: dict,
) -> dict:
    """
    Agent 3: Final editorial review, proposing specific revisions.
    Returns revision dict with quality grade.
    """
    user_msg = (
        f"ORIGINAL KOREAN TEXT:\n{original_korean}\n\n"
        f"CURRENT ENGLISH TRANSLATION:\n{english_translation}\n\n"
        f"BACK-TRANSLATION QA ISSUES:\n{json.dumps(back_translation_issues, ensure_ascii=False, indent=2)}\n\n"
        f"TONE/REGISTER QA SCORES:\n{json.dumps(tone_scores, ensure_ascii=False, indent=2)}"
    )
    raw = _api_call(client, config.EDITORIAL_SYSTEM_PROMPT, user_msg, max_tokens=8192)

    try:
        result = _parse_json_response(raw)
    except json.JSONDecodeError:
        logger.warning("Agent 3: failed to parse editorial JSON")
        result = {
            "revisions": [],
            "accepted_as_is": [],
            "chunk_quality_grade": "C",
        }

    return result


def apply_revisions(translation: str, revisions: list) -> str:
    """Apply editorial revisions to a translation."""
    revised = translation
    for rev in revisions:
        original = rev.get("original_translation", "")
        replacement = rev.get("revised_translation", "")
        if original and replacement and original in revised:
            revised = revised.replace(original, replacement, 1)
            logger.info(f"Applied revision: '{original[:50]}...' -> '{replacement[:50]}...'")
    return revised


def retranslate_chunk(
    client: anthropic.Anthropic,
    korean_text: str,
    current_translation: str,
    revisions: list,
    glossary: dict,
    chunk_num: int,
) -> str:
    """Re-translate a chunk with editorial corrections folded into the prompt."""
    revision_context = "\n".join(
        f"- Change '{r.get('original_translation', '')}' to '{r.get('revised_translation', '')}' "
        f"because: {r.get('reasoning', '')}"
        for r in revisions
    )

    glossary_str = "\n".join(f"  {k} → {v}" for k, v in glossary.items()) if glossary else "(none)"

    user_msg = (
        f"GLOSSARY:\n{glossary_str}\n\n"
        f"The following corrections were identified in the previous translation. "
        f"Please incorporate these corrections while re-translating:\n{revision_context}\n\n"
        f"PREVIOUS TRANSLATION (for reference, with known issues):\n{current_translation}\n\n"
        f"KOREAN TEXT TO RE-TRANSLATE:\n{korean_text}"
    )

    raw = _api_call(client, config.TRANSLATION_SYSTEM_PROMPT, user_msg, max_tokens=8192)
    logger.info(f"Chunk {chunk_num}: re-translated with corrections")
    return raw


def validate_chunk(
    client: anthropic.Anthropic,
    korean_text: str,
    english_translation: str,
    chunk_num: int,
    total_chunks: int,
    glossary: dict,
) -> dict:
    """
    Run all three validation agents on a single chunk.
    If quality is below threshold, attempt re-translation.
    Returns validation result dict.
    """
    print(f"Validating chunk {chunk_num}/{total_chunks}... ", end="", flush=True)

    # Run Agent 1: Back-translation check
    logger.info(f"Chunk {chunk_num}: running back-translation check")
    bt_issues = agent1_back_translation(client, korean_text, english_translation)
    time.sleep(config.RATE_LIMIT_DELAY)

    # Run Agent 2: Tone & register check
    logger.info(f"Chunk {chunk_num}: running tone check")
    tone_result = agent2_tone_check(client, korean_text, english_translation)
    time.sleep(config.RATE_LIMIT_DELAY)

    # Run Agent 3: Editorial reconciliation
    logger.info(f"Chunk {chunk_num}: running editorial review")
    editorial = agent3_editorial(client, korean_text, english_translation, bt_issues, tone_result)
    time.sleep(config.RATE_LIMIT_DELAY)

    # Check if re-translation is needed
    overall_score = tone_result.get("overall_score", 5.0)
    has_high_severity = any(
        issue.get("severity") == "HIGH" for issue in bt_issues
    )
    needs_retranslation = (
        overall_score < config.VALIDATION_THRESHOLD or has_high_severity
    )

    final_translation = english_translation
    retranslation_attempts = 0

    while needs_retranslation and retranslation_attempts < config.MAX_RETRANSLATION_ATTEMPTS:
        retranslation_attempts += 1
        logger.info(
            f"Chunk {chunk_num}: re-translating (attempt {retranslation_attempts}, "
            f"score={overall_score}, high_severity={has_high_severity})"
        )
        print(f"re-translating (attempt {retranslation_attempts})... ", end="", flush=True)

        # Apply revisions and re-translate
        revisions = editorial.get("revisions", [])
        final_translation = retranslate_chunk(
            client, korean_text, final_translation, revisions, glossary, chunk_num
        )
        time.sleep(config.RATE_LIMIT_DELAY)

        # Re-validate the new translation
        bt_issues = agent1_back_translation(client, korean_text, final_translation)
        time.sleep(config.RATE_LIMIT_DELAY)
        tone_result = agent2_tone_check(client, korean_text, final_translation)
        time.sleep(config.RATE_LIMIT_DELAY)
        editorial = agent3_editorial(client, korean_text, final_translation, bt_issues, tone_result)
        time.sleep(config.RATE_LIMIT_DELAY)

        overall_score = tone_result.get("overall_score", 5.0)
        has_high_severity = any(issue.get("severity") == "HIGH" for issue in bt_issues)
        needs_retranslation = overall_score < config.VALIDATION_THRESHOLD or has_high_severity

    # Flag for human review if still below threshold after max attempts
    needs_human_review = needs_retranslation

    result = {
        "chunk_num": chunk_num,
        "back_translation_issues": bt_issues,
        "tone_scores": tone_result,
        "editorial": editorial,
        "final_translation": final_translation,
        "retranslation_attempts": retranslation_attempts,
        "needs_human_review": needs_human_review,
        "quality_grade": editorial.get("chunk_quality_grade", "C"),
    }

    status = "NEEDS HUMAN REVIEW" if needs_human_review else "done"
    print(f"{status} (grade: {result['quality_grade']})")

    return result


def run_validation() -> dict:
    """
    Run validation on all translated chunks.
    Returns summary dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    full_korean_path = os.path.join(config.KOREAN_TEXT_DIR, "full_korean_text.txt")
    if not os.path.exists(full_korean_path):
        logger.error("full_korean_text.txt not found. Run OCR first.")
        return {"error": "OCR output not found"}

    with open(full_korean_path, "r", encoding="utf-8") as f:
        full_korean = f.read()

    # Import chunk splitter from translate module
    from translate import split_into_chunks
    chunks = split_into_chunks(full_korean)
    total = len(chunks)

    # Load glossary
    glossary = {}
    if os.path.exists(config.GLOSSARY_PATH):
        with open(config.GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)

    results = []
    human_review_chunks = []

    for chunk in chunks:
        num = chunk["chunk_num"]
        validation_path = os.path.join(config.VALIDATION_DIR, f"chunk_{num:03d}_validation.json")

        # Resume support
        if os.path.exists(validation_path):
            logger.info(f"Chunk {num}: validation already exists, skipping")
            with open(validation_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            results.append(result)
            if result.get("needs_human_review"):
                human_review_chunks.append(num)
            continue

        # Load the translation for this chunk
        translation_path = os.path.join(config.TRANSLATIONS_DIR, f"chunk_{num:03d}.txt")
        if not os.path.exists(translation_path):
            logger.error(f"Chunk {num}: translation not found, skipping validation")
            continue

        with open(translation_path, "r", encoding="utf-8") as f:
            english_translation = f.read()

        result = validate_chunk(
            client, chunk["text"], english_translation, num, total, glossary
        )
        results.append(result)

        # Save the validated/revised translation back
        if result["final_translation"] != english_translation:
            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(result["final_translation"])
            logger.info(f"Chunk {num}: updated translation with revisions")

        # Save validation report (without the full translation text to save space)
        report = {k: v for k, v in result.items() if k != "final_translation"}
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if result.get("needs_human_review"):
            human_review_chunks.append(num)

    summary = {
        "total_chunks": total,
        "validated": len(results),
        "human_review_needed": human_review_chunks,
        "grades": {r.get("chunk_num", "?"): r.get("quality_grade", "?") for r in results},
    }

    print(f"\nValidation complete. {len(results)}/{total} chunks validated.")
    if human_review_chunks:
        print(f"Chunks needing human review: {human_review_chunks}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_validation()
