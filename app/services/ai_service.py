import json
import logging
import re
from typing import Any, List, Optional

from fastapi import HTTPException
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)

SUPPORTED_HOOK_PLATFORMS = {
    "youtube": "YouTube",
    "youtube shorts": "YouTube Shorts",
    "instagram": "Instagram",
    "instagram reels": "Instagram Reels",
    "tiktok": "TikTok",
    "x": "X",
    "twitter": "X",
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
}

SUPPORTED_HUMANIZER_PLATFORMS = {
    "general": "General",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "x": "X",
    "twitter": "X",
    "linkedin": "LinkedIn",
    "youtube": "YouTube",
    "whatsapp": "WhatsApp",
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
}

HOOK_TEMPLATE_GUIDANCE = {
    "curiosity": "Prioritize curiosity gaps, intrigue, incomplete information, and a reason to keep watching.",
    "authority": "Prioritize expertise, certainty, confidence, and credible insight.",
    "problem": "Prioritize pain points, mistakes, bottlenecks, and urgency.",
    "benefit": "Prioritize outcomes, transformation, wins, and useful payoff.",
    "controversy": "Prioritize bold takes, tension, contrarian framing, and pattern interruption.",
    "story": "Prioritize narrative setup, emotional tension, and human relatability.",
}

HOOK_FAMILIES = [
    "misconception",
    "tradeoff",
    "hidden_cost",
    "framework",
    "false_assumption",
    "warning",
]

TITLE_ARCHETYPES = [
    "search_led",
    "contrarian",
    "mistake_led",
    "result_led",
    "curiosity_led",
    "proof_led",
]

CAPTION_STYLES = [
    "story",
    "authority",
    "community",
    "urgency",
]

SCRIPT_STYLES = [
    "direct_to_camera",
    "story_led",
    "problem_solution",
    "myth_busting",
]

GENERIC_HOOK_PATTERNS = [
    r"\bone will surprise you\b",
    r"\bnobody talks about\b",
    r"\bwhat actually works\b",
    r"\bbefore you post about\b",
    r"\bthe truth about\b",
    r"\bthis one tiny habit\b",
    r"\bhere'?s exactly what did\b",
    r"\bignore this\b",
    r"\bthis planning step\b",
    r"\bchanged everything\b",
]

EXAGGERATED_HOOK_PATTERNS = [
    r"\bmade me financially independent by \d+\b",
    r"\bmade me rich\b",
    r"\bchanged everything\b",
    r"\bsecret nobody tells you\b",
    r"\bguaranteed\b",
    r"\binstantly\b",
    r"\bovernight\b",
    r"\balways works\b",
    r"\bnever fails\b",
]

FIRST_PERSON_PATTERNS = [
    r"\bi used to\b",
    r"\bi thought\b",
    r"\bi learned\b",
    r"\bi realized\b",
    r"\bmy biggest mistake\b",
    r"\bhere'?s what changed for me\b",
    r"\bi went from\b",
]

WEAK_WARNING_PATTERNS = [
    r"\bignore this\b",
    r"\bstop doing this\b",
    r"\bwarning:\b",
    r"\bdon't do this\b",
]

CLICKBAIT_WORDS = {
    "shocking",
    "secret",
    "insane",
    "crazy",
    "wild",
    "brutal",
    "destroy",
    "explode",
    "instantly",
    "overnight",
}

BELIEVABILITY_WORDS = {
    "framework",
    "mistake",
    "review",
    "habit",
    "system",
    "question",
    "reason",
    "signal",
    "shift",
    "lesson",
    "decision",
    "pattern",
    "number",
    "rule",
    "cost",
    "tradeoff",
    "assumption",
    "timeline",
    "cashflow",
}

YOUTUBE_HOOK_WORDS = {
    "why",
    "how",
    "mistake",
    "truth",
    "reason",
    "question",
    "signal",
    "shift",
    "cost",
    "review",
    "numbers",
    "tradeoff",
    "assumption",
    "timeline",
}

SHORTFORM_HOOK_WORDS = {
    "stop",
    "most people",
    "this is why",
    "here's why",
    "warning",
    "confession",
}

REALISM_BAD_PHRASES = [
    r"\bmade me financially independent by \d+\b",
    r"\bi went from .* to .* freedom\b",
    r"\bchanged everything\b",
    r"\bexact spreadsheet i used\b",
    r"\bsecret nobody tells you\b",
    r"\bthis one tiny habit\b",
    r"\bguaranteed\b",
    r"\bovernight\b",
    r"\binstantly\b",
]

GENERIC_MARKETING_PHRASES = [
    r"\bjoin hundreds of people\b",
    r"\blet'?s build momentum together\b",
    r"\breader wins\b",
    r"\bwatch till the end\b",
    r"\bexact spreadsheet\b",
    r"\bwhat actually works\b",
    r"\bthis changes everything\b",
    r"\bjoin a growing community\b",
    r"\bstick around for\b",
    r"\bready to make a real change\b",
    r"\bthis is how momentum builds\b",
    r"\bmeet three real-life\b",
]

COMMUNITY_TEMPLATE_PHRASES = [
    r"\bjoin a growing community\b",
    r"\bdrop a question in the comments\b",
    r"\bthis is how momentum builds\b",
    r"\blet others weigh in\b",
]

BROAD_MOTIVATION_PHRASES = [
    r"\bready to make a real change\b",
    r"\bdon'?t wait for the right time\b",
    r"\bsmall, consistent moves\b",
    r"\bfeel possible[-—]not impossible\b",
    r"\bstick around for\b",
]

AI_TELL_PATTERNS = [
    r"\bin today'?s (video|post|thread)\b",
    r"\bdelve into\b",
    r"\bleverage\b",
    r"\bunlock\b",
    r"\belevate\b",
    r"\bgame[- ]changer\b",
    r"\bstand out in today'?s\b",
    r"\bmore than ever\b",
    r"\btapestry\b",
    r"\bembark on\b",
    r"\bjourney\b",
    r"\bpowerful\b",
    r"\btransform\b",
    r"\bseamless\b",
    r"\bnot only\b.*\bbut also\b",
]

HUMANIZER_TONES = {
    "natural": "Natural",
    "conversational": "Conversational",
    "bold": "Bold",
    "professional": "Professional",
    "warm": "Warm",
    "confident": "Confident",
    "persuasive": "Persuasive",
    "story-driven": "Story-driven",
}

HUMANIZER_STRENGTHS = {
    "light": "Light",
    "balanced": "Balanced",
    "strong": "Strong",
}


def _ensure_openai_ready() -> None:
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is missing. Add OPENAI_API_KEY to your backend environment."
        )


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clean_multiline(value: str | None) -> str:
    value = value or ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _safe_json_load(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1).strip())

        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        raise


def _obj_get(value: Any, key: str, default=None):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_debug_shape(response: Any) -> str:
    try:
        output = _obj_get(response, "output", [])
        parts = []
        for item in output[:3]:
            item_type = _obj_get(item, "type", type(item).__name__)
            content = _obj_get(item, "content", []) or []
            content_types = []
            for c in content[:5]:
                content_types.append(str(_obj_get(c, "type", type(c).__name__)))
            parts.append(f"{item_type}:{content_types}")
        return f"output_types={parts}"
    except Exception:
        return "output_types=<unavailable>"


def _extract_output_text(response: Any) -> str:
    output_text = _obj_get(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _obj_get(response, "output", []) or []
    collected: list[str] = []

    for item in output:
        content = _obj_get(item, "content", []) or []

        for part in content:
            part_type = _obj_get(part, "type", None)

            if part_type == "output_text":
                text_value = _obj_get(part, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value.strip())
                    continue

            text_value = _obj_get(part, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                collected.append(text_value.strip())
                continue

            nested_text = _obj_get(text_value, "value", None)
            if isinstance(nested_text, str) and nested_text.strip():
                collected.append(nested_text.strip())
                continue

        direct_text = _obj_get(item, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            collected.append(direct_text.strip())

    if collected:
        return "\n".join(collected).strip()

    logger.error("Model returned no extractable text. %s", _response_debug_shape(response))
    raise ValueError("Model returned no text output.")


def _normalize_language(language: str | None) -> str:
    cleaned = _clean_text(language or "en").lower()
    return cleaned if cleaned in SUPPORTED_LANGUAGES else "en"


def _language_name(language: str | None) -> str:
    return SUPPORTED_LANGUAGES.get(_normalize_language(language), "English")


def _normalize_platform(platform: str | None) -> str:
    cleaned = _clean_text(platform or "")
    if not cleaned:
        return "General"
    return SUPPORTED_HOOK_PLATFORMS.get(cleaned.lower(), cleaned)


def _normalize_humanizer_platform(platform: str | None) -> str:
    cleaned = _clean_text(platform or "").lower()
    if not cleaned:
        return "General"
    return SUPPORTED_HUMANIZER_PLATFORMS.get(cleaned, platform.strip() if platform else "General")


def _normalize_humanizer_tone(tone: str | None) -> str:
    cleaned = _clean_text(tone or "Natural").lower()
    return HUMANIZER_TONES.get(cleaned, tone.strip() if tone else "Natural")


def _normalize_humanizer_strength(strength: str | None) -> str:
    cleaned = _clean_text(strength or "Balanced").lower()
    return HUMANIZER_STRENGTHS.get(cleaned, strength.strip() if strength else "Balanced")


def _platform_hook_profile(platform: str) -> dict[str, Any]:
    platform = _normalize_platform(platform).lower()

    if platform == "youtube":
        return {
            "style": "title-like but still hooky",
            "word_range": "8 to 15 words",
            "priority": "clarity, curiosity, usefulness, believable tension",
            "avoid": "cheap clickbait, fake confession framing, vague warnings, over-specific unsupported claims",
        }

    if platform in {"youtube shorts", "instagram reels", "tiktok"}:
        return {
            "style": "spoken and punchy",
            "word_range": "6 to 12 words",
            "priority": "fast stop power, natural spoken rhythm, clean contrast",
            "avoid": "bloated title phrasing, fake guru energy, generic warnings",
        }

    if platform in {"x", "linkedin", "facebook", "instagram"}:
        return {
            "style": "social-native opener",
            "word_range": "7 to 16 words",
            "priority": "relatability, tension, clarity, conversational feel",
            "avoid": "spammy curiosity bait and overpromises",
        }

    return {
        "style": "clear and curiosity-led",
        "word_range": "6 to 14 words",
        "priority": "specificity, relevance, believable tension",
        "avoid": "generic template language",
    }


def _parse_csvish(value: str | None) -> list[str]:
    if not value:
        return []

    parts = re.split(r"[,\n|;/]+", value)
    cleaned: list[str] = []
    seen: set[str] = set()

    for part in parts:
        item = _clean_text(part)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)

    return cleaned


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def _dedupe_text_items(
    items: list[str],
    *,
    limit: int,
    min_items: int = 1,
    similarity_threshold: float = 0.72,
) -> list[str]:
    cleaned: list[str] = []
    seen_keys: set[str] = set()

    for item in items:
        text = _clean_text(item)
        if not text:
            continue

        key = _normalized_key(text)
        if key in seen_keys:
            continue

        too_similar = any(_similarity(text, existing) >= similarity_threshold for existing in cleaned)
        if too_similar:
            continue

        seen_keys.add(key)
        cleaned.append(text)

        if len(cleaned) >= limit:
            break

    if len(cleaned) < min_items:
        raise ValueError("Not enough distinct items returned.")

    return cleaned


def _context_summary(context: dict[str, Any]) -> str:
    lines: list[str] = []

    def add(label: str, value: Any):
        if isinstance(value, str):
            cleaned = _clean_text(value)
            if cleaned:
                lines.append(f"- {label}: {cleaned}")
        elif isinstance(value, list):
            cleaned_list = [str(v).strip() for v in value if str(v).strip()]
            if cleaned_list:
                lines.append(f"- {label}: {', '.join(cleaned_list)}")
        elif isinstance(value, bool):
            lines.append(f"- {label}: {'Yes' if value else 'No'}")

    add("Topic", context.get("topic"))
    add("Platform", context.get("platform"))
    add("Audience", context.get("audience"))
    add("Goal", context.get("goal"))
    add("Offer", context.get("offer"))
    add("Pain point", context.get("pain_point"))
    add("Desired outcome", context.get("desired_outcome"))
    add("Angle", context.get("angle"))
    add("CTA", context.get("call_to_action"))
    add("Tone", context.get("tone"))
    add("Brand voice", context.get("brand_voice"))
    add("Keywords", context.get("keywords"))
    add("Content type", context.get("content_type"))
    add("Template", context.get("template"))
    add("Original description", context.get("original_description"))
    add("Thumbnail text", context.get("thumbnail_text"))
    add("Avoid phrases", context.get("avoid_phrases"))
    add("Extra context", context.get("extra_context"))
    add("Original text", context.get("original_text"))
    add("Humanization strength", context.get("humanization_strength"))
    add("Preserve original meaning", context.get("preserve_original_meaning"))
    add("Style notes", context.get("style_notes"))

    return "\n".join(lines).strip()


def _responses_create(*, instructions: str, input_text: str, max_output_tokens: int = 1200) -> Any:
    _ensure_openai_ready()

    json_input_text = f"""
Return valid JSON only.

{input_text}
""".strip()

    return client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json_input_text,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "minimal"},
    )


def _call_json_model(
    *,
    instructions: str,
    prompt: str,
    max_output_tokens: int = 1200,
) -> Any:
    try:
        response = _responses_create(
            instructions=instructions,
            input_text=prompt,
            max_output_tokens=max_output_tokens,
        )
        raw_text = _extract_output_text(response)

        try:
            return _safe_json_load(raw_text)
        except Exception as first_parse_error:
            logger.warning("Initial JSON parse failed. Attempting repair. Error: %s", str(first_parse_error))

            repair_instructions = """
You are a JSON repair assistant.
Convert the provided content into valid JSON only.
Do not add commentary.
Do not wrap in markdown.
Return only syntactically valid JSON.
"""
            repair_prompt = f"""
Return valid JSON only.

Convert this into valid JSON only.

RAW CONTENT:
{raw_text}
"""
            repair_response = _responses_create(
                instructions=repair_instructions,
                input_text=repair_prompt,
                max_output_tokens=max_output_tokens,
            )
            repaired_text = _extract_output_text(repair_response)
            return _safe_json_load(repaired_text)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenAI request failed.")
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI request failed: {str(exc)}"
        ) from exc


def _title_label(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Decent"
    return "Weak"


def _postprocess_hook_text(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"\s+([?!.,:;])", r"\1", text)
    text = re.sub(r"\bhere'?s exactly what did\b", "here’s what changed", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwarning:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _postprocess_general_text(text: str) -> str:
    text = _clean_multiline(text)
    text = re.sub(r"\bwatch till the end\b", "stay with this", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis changes everything\b", "this matters more than most people think", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstick around for\b", "at the end, you'll get", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjoin a growing community\b", "learn alongside other people working on the same goal", text, flags=re.IGNORECASE)
    return text.strip()


def _postprocess_humanizer_text(text: str) -> str:
    text = _clean_multiline(text)
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _supports_first_person_context(
    *,
    offer: str | None = None,
    extra_context: str | None = None,
    angle: str | None = None,
    brand_voice: str | None = None,
) -> bool:
    combined = " ".join(
        [
            _clean_text(offer),
            _clean_text(extra_context),
            _clean_text(angle),
            _clean_text(brand_voice),
        ]
    ).lower()

    support_signals = [
        "personal story",
        "my story",
        "founder story",
        "confession",
        "my experience",
        "my lesson",
        "case study",
        "journey",
        "behind the scenes",
    ]
    return any(signal in combined for signal in support_signals)


def _has_unrealistic_claims(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in REALISM_BAD_PHRASES)


def _has_generic_marketing(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in GENERIC_MARKETING_PHRASES)


def _has_community_template(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in COMMUNITY_TEMPLATE_PHRASES)


def _has_broad_motivation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in BROAD_MOTIVATION_PHRASES)


def _has_ai_tell(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in AI_TELL_PATTERNS)


def _score_hook_text(text: str, platform: str = "General", support_first_person: bool = False) -> int:
    cleaned = _clean_text(text)
    words = [w for w in cleaned.split(" ") if w]
    word_count = len(words)
    char_count = len(cleaned)
    lowered = cleaned.lower()

    score = 52

    platform_lower = _normalize_platform(platform).lower()
    if platform_lower == "youtube":
        if 8 <= word_count <= 15:
            score += 14
        elif 6 <= word_count <= 17:
            score += 8
        else:
            score -= 4
    elif platform_lower in {"youtube shorts", "instagram reels", "tiktok"}:
        if 5 <= word_count <= 11:
            score += 14
        elif 4 <= word_count <= 13:
            score += 8
        else:
            score -= 4
    else:
        if 6 <= word_count <= 14:
            score += 12
        elif 5 <= word_count <= 16:
            score += 7
        else:
            score -= 3

    if char_count <= 90:
        score += 10
    elif char_count <= 120:
        score += 5
    else:
        score -= 4

    if any(word in lowered for word in BELIEVABILITY_WORDS):
        score += 12

    if platform_lower == "youtube":
        if any(word in lowered for word in YOUTUBE_HOOK_WORDS):
            score += 8
    elif platform_lower in {"youtube shorts", "instagram reels", "tiktok"}:
        if any(word in lowered for word in SHORTFORM_HOOK_WORDS):
            score += 8

    if re.search(r"\b(why|how|mistake|reason|tradeoff|assumption|cost|question|signal|framework)\b", lowered):
        score += 10

    if re.search(r"\b(isn’t enough|isn't enough|what most people miss|get wrong|quietly costs|slows down|keeps you from)\b", lowered):
        score += 10

    if any(re.search(pattern, lowered) for pattern in GENERIC_HOOK_PATTERNS):
        score -= 9

    if any(re.search(pattern, lowered) for pattern in EXAGGERATED_HOOK_PATTERNS):
        score -= 12

    if not support_first_person and any(re.search(pattern, lowered) for pattern in FIRST_PERSON_PATTERNS):
        score -= 12

    if any(re.search(pattern, lowered) for pattern in WEAK_WARNING_PATTERNS):
        score -= 8

    clickbait_hits = sum(1 for word in CLICKBAIT_WORDS if word in lowered)
    if clickbait_hits >= 2:
        score -= 10
    elif clickbait_hits == 1:
        score -= 4

    if len(re.findall(r"\b\d+\b", cleaned)) >= 2:
        score -= 5

    if re.search(r"\b(review|rule|numbers|question|mistake|habit|shift|pattern|cost|timeline|cashflow)\b", lowered):
        score += 6

    return _clamp(score, 35, 98)


def _score_title_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_text(text)
    words = len(cleaned.split())
    chars = len(cleaned)

    score = 56
    if 35 <= chars <= 68:
        score += 14
    elif 25 <= chars <= 78:
        score += 8

    if 5 <= words <= 11:
        score += 12
    elif 4 <= words <= 13:
        score += 6

    if re.search(r"\b(how|why|what|best|mistake|truth|guide|questions?)\b", cleaned, re.IGNORECASE):
        score += 10
    if re.search(r"[:?!—-]", cleaned):
        score += 5

    if re.search(r"\b(secret|insane|crazy|overnight)\b", cleaned, re.IGNORECASE):
        score -= 8

    if _has_unrealistic_claims(cleaned):
        score -= 12

    if _has_generic_marketing(cleaned):
        score -= 8

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 40, 98)


def _score_caption_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_multiline(text)
    chars = len(cleaned)
    score = 56

    if 90 <= chars <= 260:
        score += 14
    elif 50 <= chars <= 320:
        score += 8

    if "\n" in cleaned:
        score += 6
    if re.search(r"\b(comment|save|share|follow|watch|learn|try)\b", cleaned, re.IGNORECASE):
        score += 10
    if re.search(r"\b(you|your)\b", cleaned, re.IGNORECASE):
        score += 8

    if re.search(r"\b(secret|insane|crazy|overnight)\b", cleaned, re.IGNORECASE):
        score -= 6

    if _has_unrealistic_claims(cleaned):
        score -= 12

    if _has_generic_marketing(cleaned):
        score -= 8

    if _has_community_template(cleaned):
        score -= 8

    if _has_broad_motivation(cleaned):
        score -= 6

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 40, 98)


def _score_description_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_multiline(text)
    score = 58

    if len(cleaned) >= 180:
        score += 10
    if cleaned.count("\n") >= 2:
        score += 8
    if re.search(r"\b(keyword|learn|discover|watch|tips|guide)\b", cleaned, re.IGNORECASE):
        score += 10
    if re.search(r"\b(subscribe|follow|comment|visit|download)\b", cleaned, re.IGNORECASE):
        score += 8

    if _has_unrealistic_claims(cleaned):
        score -= 12

    if _has_generic_marketing(cleaned):
        score -= 8

    if _has_broad_motivation(cleaned):
        score -= 6

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 40, 98)


def _score_script_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_multiline(text)
    score = 56

    if len(cleaned) >= 160:
        score += 10
    if re.search(r"\bHook:\b", cleaned, re.IGNORECASE):
        score += 8
    if re.search(r"\bCTA:\b", cleaned, re.IGNORECASE):
        score += 8
    if re.search(r"\b(you|your)\b", cleaned, re.IGNORECASE):
        score += 8
    if "\n" in cleaned:
        score += 8

    if re.search(r"\b(secret|insane|overnight)\b", cleaned, re.IGNORECASE):
        score -= 8

    if _has_unrealistic_claims(cleaned):
        score -= 12

    if _has_generic_marketing(cleaned):
        score -= 8

    if _has_broad_motivation(cleaned):
        score -= 6

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 40, 98)


def _score_hashtag_set(tags: list[str]) -> int:
    count = len(tags)
    score = 54

    if 10 <= count <= 18:
        score += 18
    elif 7 <= count <= 20:
        score += 10

    short = sum(1 for tag in tags if len(tag) <= 12)
    medium = sum(1 for tag in tags if 13 <= len(tag) <= 20)
    if short >= 3:
        score += 8
    if medium >= 3:
        score += 8

    return _clamp(score, 40, 98)


def _score_thumbnail_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_text(text)
    words = len(cleaned.split())
    chars = len(cleaned)

    score = 52
    if 2 <= words <= 6:
        score += 18
    elif 1 <= words <= 8:
        score += 10

    if chars <= 28:
        score += 12
    elif chars <= 40:
        score += 6

    if re.search(r"\b(secret|mistake|truth|stop|why|how|real|wrong|better)\b", cleaned, re.IGNORECASE):
        score += 12

    if _has_unrealistic_claims(cleaned):
        score -= 10

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 35, 98)


def _score_humanizer_text(text: str, original_text: str, preserve_original_meaning: bool) -> int:
    cleaned = _clean_multiline(text)
    original_cleaned = _clean_multiline(original_text)

    score = 58

    if cleaned and len(cleaned) > 0:
        score += 6

    if "\n" in cleaned:
        score += 4

    if re.search(r"\b(you|your|we)\b", cleaned, re.IGNORECASE):
        score += 6

    if not _has_ai_tell(cleaned):
        score += 12
    else:
        score -= 10

    if not _has_generic_marketing(cleaned):
        score += 6
    else:
        score -= 8

    if not _has_unrealistic_claims(cleaned):
        score += 5
    else:
        score -= 10

    if cleaned.lower() != original_cleaned.lower():
        score += 6

    if preserve_original_meaning:
        similarity = _similarity(cleaned, original_cleaned)
        if similarity >= 0.45:
            score += 8
        elif similarity < 0.2:
            score -= 10

    if len(cleaned.split()) < 3:
        score -= 12

    return _clamp(score, 40, 98)


def _fallback_hooks(topic: str, platform: str, template: str) -> list[dict[str, Any]]:
    topic = _clean_text(topic)
    platform = _normalize_platform(platform)
    template = _clean_text(template or "curiosity").lower()

    bank = {
        "curiosity": [
            f"Why {topic} feels harder than it should",
            f"What people miss about {topic}",
            f"The part of {topic} nobody prepares you for",
            f"How {topic} actually starts paying off",
            f"What changes when you approach {topic} differently",
            f"The real reason most people struggle with {topic}",
        ],
        "authority": [
            f"Here’s the smarter way to approach {topic}",
            f"What strong performers understand about {topic}",
            f"The disciplined way to build {topic}",
            f"How experienced people think about {topic}",
            f"The better framework for {topic}",
            f"What separates average from strong results in {topic}",
        ],
        "problem": [
            f"This is why your {topic} progress feels slow",
            f"The mistake that keeps people stuck in {topic}",
            f"What quietly ruins momentum in {topic}",
            f"Why most {topic} advice doesn’t help enough",
            f"The hidden bottleneck in {topic}",
            f"What’s making {topic} harder than it needs to be",
        ],
        "benefit": [
            f"How {topic} creates more freedom over time",
            f"What better {topic} can unlock for you",
            f"Why improving {topic} changes more than you think",
            f"The upside of getting {topic} right",
            f"How {topic} can build stronger long-term results",
            f"What happens when {topic} finally clicks",
        ],
        "controversy": [
            f"Most people are overthinking {topic}",
            f"The common advice on {topic} is too shallow",
            f"Why I disagree with how people teach {topic}",
            f"Not everyone should approach {topic} the same way",
            f"The uncomfortable truth about {topic}",
            f"What people get wrong about {topic}",
        ],
        "story": [
            f"I used to misunderstand {topic}",
            f"What changed my mind about {topic}",
            f"The moment {topic} started making sense",
            f"What one lesson taught me about {topic}",
            f"How my view on {topic} completely changed",
            f"The turning point in understanding {topic}",
        ],
    }

    selected = bank.get(template, bank["curiosity"])
    return [
        {
            "text": text,
            "score": _score_hook_text(text, platform),
            "platform": platform,
            "family": template,
        }
        for text in selected[:6]
    ]


def _fallback_titles(topic: str) -> list[dict[str, Any]]:
    topic = _clean_text(topic)
    items = [
        f"{topic}: What Most People Get Wrong",
        f"How to Make {topic} Work Better",
        f"The Smarter Way to Approach {topic}",
        f"{topic}: Small Changes That Drive Bigger Results",
        f"Why Your {topic} Strategy Feels Stuck",
        f"{topic}: A Better Place to Start",
    ]
    return [
        {"text": item, "score": _score_title_text(item), "label": _title_label(_score_title_text(item))}
        for item in items
    ]


def _fallback_captions(topic: str, platform: str, tone: str) -> list[dict[str, Any]]:
    topic = _clean_text(topic)
    platform = _normalize_platform(platform)
    tone = _clean_text(tone or "engaging")

    items = [
        f"Most people treat {topic} like a small detail, but on {platform}, it changes how your content gets perceived.\n\nThis is where a more {tone} angle starts to win.\n\nSave this if you want better content decisions.",
        f"If your {topic} feels repetitive, the problem usually is not effort. It is weak positioning.\n\nA stronger angle makes people stop, read, and care.\n\nComment if you want more.",
        f"Better {topic} is rarely about doing more. It is about saying the right thing in a sharper way.\n\nThat shift matters more on {platform} than most creators realize.\n\nShare this with someone building content right now.",
        f"Here is the truth: {topic} can feel average or magnetic depending on how you frame it.\n\nThe difference is clarity, tension, and relevance.\n\nFollow for more practical content strategy.",
    ]

    return [
        {"text": item, "score": _score_caption_text(item), "label": _title_label(_score_caption_text(item))}
        for item in items
    ]


def _fallback_hashtags(topic: str, platform: str) -> list[dict[str, Any]]:
    topic_token = re.sub(r"[^a-zA-Z0-9]+", "", topic.title()) or "Content"
    platform_token = re.sub(r"[^a-zA-Z0-9]+", "", platform.title()) or "Social"

    sets = [
        {
            "title": "Broad Discovery Set",
            "description": "Balanced for general discoverability.",
            "tags": [
                f"#{topic_token}",
                "#ContentStrategy",
                "#CreatorTips",
                "#GrowthTips",
                "#DigitalMarketing",
                "#SocialMediaTips",
                f"#{platform_token}",
                "#OnlineGrowth",
                "#AudienceGrowth",
                "#MarketingIdeas",
                "#BrandGrowth",
                "#ContentCreation",
            ],
        },
        {
            "title": "Niche Relevance Set",
            "description": "Tighter intent for more targeted reach.",
            "tags": [
                f"#{topic_token}Tips",
                f"#{topic_token}Strategy",
                f"#{topic_token}Growth",
                "#CreatorEducation",
                "#ContentSystems",
                "#AudienceBuilding",
                "#OrganicReach",
                "#ContentIdeas",
                "#CreatorBusiness",
                "#StrategicContent",
            ],
        },
        {
            "title": "Engagement Set",
            "description": "Useful when the goal is saves, shares, and discussion.",
            "tags": [
                f"#{topic_token}",
                "#SaveThisPost",
                "#LearnWithMe",
                "#CreatorMindset",
                "#ActionableTips",
                "#ContentBreakdown",
                "#MarketingLessons",
                "#GrowthMindset",
                "#ContentWins",
                "#BuildInPublic",
            ],
        },
    ]

    normalized = []
    for item in sets:
        score = _score_hashtag_set(item["tags"])
        normalized.append({**item, "score": score, "label": _title_label(score)})
    return normalized


def _fallback_descriptions(topic: str, platform: str) -> list[dict[str, Any]]:
    platform = _normalize_platform(platform)
    items = [
        {
            "title": "SEO-Focused",
            "text": (
                f"If you want better results with {topic}, this breakdown will help you approach it more strategically on {platform}.\n\n"
                f"In this piece, you will learn what weakens {topic}, what makes it more effective, and how to improve clarity, positioning, and performance.\n\n"
                "Watch, save, and share if you want more practical creator growth tips."
            ),
        },
        {
            "title": "Audience-Focused",
            "text": (
                f"Struggling to make {topic} feel stronger, clearer, or more valuable on {platform}?\n\n"
                f"This version focuses on what your audience actually notices: relevance, simplicity, and strong payoff.\n\n"
                "If that sounds useful, stay with this and apply one idea today."
            ),
        },
        {
            "title": "CTA-Focused",
            "text": (
                f"{topic} can perform a lot better when the framing is sharper.\n\n"
                f"This rewrite is designed to make your {platform} content more specific, more engaging, and easier for viewers to act on.\n\n"
                "Save this for later, send it to a creator friend, and follow for more."
            ),
        },
    ]

    result = []
    for item in items:
        score = _score_description_text(item["text"])
        result.append({**item, "score": score, "label": _title_label(score), "platform": platform})
    return result


def _fallback_scripts(topic: str, platform: str) -> list[dict[str, Any]]:
    platform = _normalize_platform(platform)
    drafts = [
        {
            "title": "Direct Hook Script",
            "style": "direct-to-camera",
            "hook": f"If your {topic} feels weak, this is probably why.",
            "body": f"Most people think {topic} is just about effort. It is not. The real difference comes from sharper framing, clearer payoff, and better audience relevance.",
            "cta": "Save this and follow for more creator strategy.",
        },
        {
            "title": "Problem-Solution Script",
            "style": "problem-solution",
            "hook": f"Here is the mistake most people make with {topic}.",
            "body": f"They focus on volume before positioning. But when your angle is weak, more content will not fix it. Start by making the message sharper and more useful.",
            "cta": "Comment if you want a part two.",
        },
        {
            "title": "Myth-Busting Script",
            "style": "myth-busting",
            "hook": f"People keep saying {topic} is easy. It is not.",
            "body": f"What makes {topic} work is not luck. It is clarity, tension, and knowing exactly why someone should care in the first few seconds.",
            "cta": "Share this with someone building content right now.",
        },
    ]

    normalized = []
    for item in drafts:
        text = f"Hook: {item['hook']}\n\nBody: {item['body']}\n\nCTA: {item['cta']}"
        score = _score_script_text(text)
        normalized.append({**item, "text": text, "score": score, "label": _title_label(score), "platform": platform})
    return normalized


def _fallback_thumbnail(topic: str, thumbnail_text: str) -> dict[str, Any]:
    text = _clean_text(thumbnail_text or topic or "Your Text")
    suggestions = [
        "The Real Reason",
        "This Changes Everything",
        "Most People Get This Wrong",
        "Stop Doing This",
        "The Better Way",
    ]
    return {
        "text": text,
        "score": _score_thumbnail_text(text),
        "label": _title_label(_score_thumbnail_text(text)),
        "feedback": [
            "Keep thumbnail text short enough to read instantly.",
            "Use stronger tension, contrast, or curiosity words.",
            "Avoid saying the exact same thing as the title.",
        ],
        "metrics": {
            "word_count": len(text.split()),
            "character_count": len(text),
        },
        "suggestions": [
            {
                "text": item,
                "score": _score_thumbnail_text(item),
                "label": _title_label(_score_thumbnail_text(item)),
            }
            for item in suggestions
        ],
    }


def _fallback_humanizer_rewrites(
    original_text: str,
    tone: str,
    platform: str,
    preserve_original_meaning: bool,
) -> list[dict[str, Any]]:
    base = _clean_multiline(original_text)
    if not base:
        raise ValueError("Original text is required.")

    platform_name = _normalize_humanizer_platform(platform)
    tone_name = _normalize_humanizer_tone(tone)

    rewrites = [
        {
            "version_name": "Best Rewrite",
            "text": base,
            "why": f"Cleaner and more natural wording for {platform_name} with a {tone_name.lower()} feel.",
        },
        {
            "version_name": "Alternative 1",
            "text": re.sub(r"\butilize\b", "use", base, flags=re.IGNORECASE),
            "why": "Simplifies stiff wording and makes the message easier to read.",
        },
        {
            "version_name": "Alternative 2",
            "text": re.sub(r"\bleverage\b", "use", base, flags=re.IGNORECASE),
            "why": "Reduces formal AI-sounding language and feels more human.",
        },
        {
            "version_name": "Alternative 3",
            "text": re.sub(r"\bin order to\b", "to", base, flags=re.IGNORECASE),
            "why": "Cuts unnecessary phrasing and improves flow.",
        },
    ]

    normalized = []
    seen = []
    for item in rewrites:
        text = _postprocess_humanizer_text(item["text"])
        if not text:
            continue
        if any(_similarity(text, existing) >= 0.9 for existing in seen):
            continue
        seen.append(text)
        score = _score_humanizer_text(text, original_text, preserve_original_meaning)
        normalized.append(
            {
                "version_name": item["version_name"],
                "text": text,
                "why": item["why"],
                "score": score,
                "label": _title_label(score),
            }
        )

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:4]


def _normalize_hook_items(
    hooks_raw: Any,
    *,
    normalized_platform: str,
    banned_texts: list[str],
    template_key: str,
    support_first_person: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    accepted_texts: list[str] = []

    for item in hooks_raw:
        if isinstance(item, dict):
            text = _postprocess_hook_text(item.get("text") or "")
            family = _clean_text(item.get("family") or template_key)
        else:
            text = _postprocess_hook_text(str(item))
            family = template_key

        if not text:
            continue
        if any(_similarity(text, banned) >= 0.72 for banned in banned_texts):
            continue
        if any(_similarity(text, existing) >= 0.72 for existing in accepted_texts):
            continue

        accepted_texts.append(text)
        normalized.append(
            {
                "text": text,
                "score": _score_hook_text(text, normalized_platform, support_first_person=support_first_person),
                "platform": normalized_platform,
                "family": family,
            }
        )
        if len(normalized) >= 6:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_title_items(items_raw: Any, *, support_first_person: bool = False) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for item in items_raw:
        if isinstance(item, dict):
            text = _clean_text(item.get("text") or "")
            archetype = _clean_text(item.get("archetype") or "general")
            keyword_focus = _clean_text(item.get("keyword_focus") or "")
        else:
            text = _clean_text(str(item))
            archetype = "general"
            keyword_focus = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.72 for existing in seen):
            continue

        seen.append(text)
        score = _score_title_text(text, support_first_person=support_first_person)
        normalized.append(
            {
                "text": text,
                "score": score,
                "label": _title_label(score),
                "archetype": archetype,
                "keyword_focus": keyword_focus,
            }
        )

        if len(normalized) >= 6:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_caption_items(items_raw: Any, *, support_first_person: bool = False) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for item in items_raw:
        if isinstance(item, dict):
            text = _postprocess_general_text(item.get("text") or "")
            style = _clean_text(item.get("style") or "general")
            cta = _clean_text(item.get("cta") or "")
        else:
            text = _postprocess_general_text(str(item))
            style = "general"
            cta = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.72 for existing in seen):
            continue

        seen.append(text)
        score = _score_caption_text(text, support_first_person=support_first_person)
        normalized.append(
            {
                "text": text,
                "score": score,
                "label": _title_label(score),
                "style": style,
                "cta": cta,
            }
        )

        if len(normalized) >= 4:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_hashtag_sets(items_raw: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = re.split(r"[\s,]+", tags)

        cleaned_tags = []
        seen = set()
        for tag in tags:
            safe_tag = _clean_text(str(tag))
            if not safe_tag:
                continue
            if not safe_tag.startswith("#"):
                safe_tag = f"#{safe_tag.lstrip('#')}"
            if len(safe_tag) < 2:
                continue
            key = safe_tag.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_tags.append(safe_tag)

        if len(cleaned_tags) < 6:
            continue

        score = _score_hashtag_set(cleaned_tags)
        normalized.append(
            {
                "title": _clean_text(item.get("title") or f"Hashtag Set {index + 1}"),
                "description": _clean_text(item.get("description") or "Strategic hashtag mix."),
                "tags": cleaned_tags[:18],
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_description_items(
    items_raw: Any,
    platform: str,
    *,
    support_first_person: bool = False,
) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for item in items_raw:
        if isinstance(item, dict):
            title = _clean_text(item.get("title") or "Rewrite")
            text = _postprocess_general_text(item.get("text") or "")
        else:
            title = "Rewrite"
            text = _postprocess_general_text(str(item))

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.75 for existing in seen):
            continue

        seen.append(text)
        score = _score_description_text(text, support_first_person=support_first_person)
        normalized.append(
            {
                "title": title,
                "text": text,
                "score": score,
                "label": _title_label(score),
                "platform": platform,
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_script_items(
    items_raw: Any,
    platform: str,
    *,
    support_first_person: bool = False,
) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        hook = _clean_text(item.get("hook") or "")
        body = _postprocess_general_text(item.get("body") or "")
        cta = _clean_text(item.get("cta") or "")
        title = _clean_text(item.get("title") or f"Script Draft {index + 1}")
        style_name = _clean_text(item.get("style") or "general")

        text = f"Hook: {hook}\n\nBody: {body}\n\nCTA: {cta}".strip()
        if not hook or not body:
            continue
        if any(_similarity(text, existing) >= 0.74 for existing in seen):
            continue

        seen.append(text)
        score = _score_script_text(text, support_first_person=support_first_person)
        normalized.append(
            {
                "title": title,
                "style": style_name,
                "hook": hook,
                "body": body,
                "cta": cta,
                "text": text,
                "score": score,
                "label": _title_label(score),
                "platform": platform,
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_thumbnail_result(
    raw: Any,
    topic: str,
    thumbnail_text: str,
    *,
    support_first_person: bool = False,
) -> dict[str, Any]:
    analysis_raw = raw.get("analysis", {}) if isinstance(raw, dict) else {}
    text_value = _clean_text(
        analysis_raw.get("text")
        or thumbnail_text
        or topic
    )
    if not text_value:
        raise ValueError("Missing thumbnail text.")

    score = _score_thumbnail_text(text_value, support_first_person=support_first_person)
    feedback = analysis_raw.get("feedback") or []
    if not isinstance(feedback, list):
        feedback = []

    suggestions_raw = raw.get("suggestions", []) if isinstance(raw, dict) else []
    suggestions = []
    seen: list[str] = []
    for item in suggestions_raw:
        if isinstance(item, dict):
            text = _clean_text(item.get("text") or "")
        else:
            text = _clean_text(str(item))

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.72 for existing in seen):
            continue

        seen.append(text)
        suggestion_score = _score_thumbnail_text(text, support_first_person=support_first_person)
        suggestions.append(
            {
                "text": text,
                "score": suggestion_score,
                "label": _title_label(suggestion_score),
            }
        )
        if len(suggestions) >= 5:
            break

    if not suggestions:
        raise ValueError("Too few thumbnail suggestions returned.")

    suggestions.sort(key=lambda item: item["score"], reverse=True)

    return {
        "text": text_value,
        "score": score,
        "label": _title_label(score),
        "feedback": [str(item).strip() for item in feedback if str(item).strip()][:5],
        "metrics": {
            "word_count": len(text_value.split()),
            "character_count": len(text_value),
        },
        "suggestions": suggestions,
    }


def _normalize_humanizer_items(
    items_raw: Any,
    *,
    original_text: str,
    preserve_original_meaning: bool,
) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if isinstance(item, dict):
            version_name = _clean_text(item.get("version_name") or item.get("title") or "")
            text = _postprocess_humanizer_text(item.get("text") or "")
            why = _clean_text(item.get("why") or "")
        else:
            version_name = ""
            text = _postprocess_humanizer_text(str(item))
            why = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.82 for existing in seen):
            continue

        seen.append(text)
        score = _score_humanizer_text(text, original_text, preserve_original_meaning)
        normalized.append(
            {
                "version_name": version_name or f"Alternative {index + 1}",
                "text": text,
                "why": why,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 4:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)

    if normalized:
        normalized[0]["version_name"] = "Best Rewrite"
        for i in range(1, len(normalized)):
            normalized[i]["version_name"] = f"Alternative {i}"

    return normalized


def _generate_with_retry(
    *,
    tool_name: str,
    primary_instructions: str,
    primary_prompt: str,
    secondary_instructions: str,
    secondary_prompt: str,
    normalize_fn,
    min_primary: int,
    min_secondary: int,
    fallback_fn,
    max_output_tokens: int = 1400,
):
    try:
        raw = _call_json_model(
            instructions=primary_instructions,
            prompt=primary_prompt,
            max_output_tokens=max_output_tokens,
        )
        normalized = normalize_fn(raw)
        count = len(normalized) if isinstance(normalized, list) else 1
        if count >= min_primary:
            logger.info("%s used primary OpenAI path successfully.", tool_name)
            return normalized
        raise ValueError(f"Primary {tool_name} generation returned too few distinct items: {count}")

    except Exception as primary_exc:
        logger.warning("Primary %s generation failed. Retrying with secondary OpenAI prompt. Reason: %s", tool_name, str(primary_exc))

        try:
            raw = _call_json_model(
                instructions=secondary_instructions,
                prompt=secondary_prompt,
                max_output_tokens=max_output_tokens,
            )
            normalized = normalize_fn(raw)
            count = len(normalized) if isinstance(normalized, list) else 1
            logger.info("Secondary %s path produced %s distinct items before final check.", tool_name, count)

            if count >= min_secondary:
                logger.info("%s used secondary OpenAI path successfully.", tool_name)
                return normalized

            raise ValueError(f"Secondary {tool_name} generation returned too few distinct items: {count}")

        except Exception as secondary_exc:
            logger.warning(
                "%s fell back to templates. Primary reason: %s | Secondary reason: %s",
                tool_name,
                str(primary_exc),
                str(secondary_exc),
            )
            return fallback_fn()


def generate_hooks(
    topic: str,
    platform: str | None = None,
    template: str | None = "curiosity",
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
    avoid_hooks: Optional[List[str]] = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    platform_profile = _platform_hook_profile(normalized_platform)
    template_key = _clean_text(template or "curiosity").lower()
    template_guidance = HOOK_TEMPLATE_GUIDANCE.get(template_key, HOOK_TEMPLATE_GUIDANCE["curiosity"])
    banned_texts = [_clean_text(item) for item in (avoid_hooks or []) if _clean_text(item)]
    support_first_person = _supports_first_person_context(
        offer=offer,
        extra_context=extra_context,
        angle=angle,
        brand_voice=brand_voice,
    )

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "angle": angle,
        "call_to_action": call_to_action,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
        "template": template_key,
    }

    first_person_rule = (
        "First-person hooks are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person confession or personal-story framing."
    )

    primary_instructions = f"""
You generate high-quality content hooks in {_language_name(language)}.
Return valid JSON only.

Platform fit:
- Platform: {normalized_platform}
- Desired style: {platform_profile['style']}
- Preferred length: {platform_profile['word_range']}
- Prioritize: {platform_profile['priority']}
- Avoid: {platform_profile['avoid']}

Rules:
- Return exactly 6 hooks under "hooks".
- Each hook must be distinctly different in structure and opening.
- Cover these families when possible: {", ".join(HOOK_FAMILIES)}.
- Push toward these angles: misconception, tradeoff, hidden cost, better framework, false assumption, grounded warning.
- Template guidance: {template_guidance}
- Make hooks feel specific to the topic, audience, and goal.
- Prefer believable and useful tension over exaggerated clickbait.
- Do not invent personal proof unless the user provided it.
- Do not make unverified claims like becoming rich, financially independent by a certain age, or overnight transformation unless clearly supported by context.
- {first_person_rule}
- Avoid vague warnings like "ignore this step" or "stop doing this" unless the missing step is named clearly.
- Avoid generic phrasing like:
  - "one will surprise you"
  - "nobody talks about"
  - "what actually works"
  - "before you post about"
  - "the truth about"
unless it is truly the strongest fit.
- Keep the writing human, sharp, and platform-native.
- Each item should be an object with:
  - text
  - family
"""

    primary_prompt = f"""
Create 6 hook options in json.

Context:
{_context_summary(context)}

Hooks or phrases to avoid:
{json.dumps(banned_texts[:20], ensure_ascii=False)}
"""

    secondary_instructions = f"""
You write strong, believable hooks in {_language_name(language)}.
Return valid JSON only.

Platform fit:
- Platform: {normalized_platform}
- Desired style: {platform_profile['style']}
- Preferred length: {platform_profile['word_range']}

Rules:
- Return exactly 6 hooks in an object under "hooks".
- Hooks must feel specific to the topic, not like templates with the topic pasted in.
- Every hook must use a different structure and opening.
- Prefer practical specificity, believable curiosity, and clean tension.
- Favor these shapes:
  1) false assumption
  2) hidden cost
  3) better framework
  4) tradeoff
  5) named mistake
  6) grounded warning
- Avoid fake guru phrasing, unsupported personal claims, generic warnings, and spammy curiosity bait.
- {first_person_rule}
- Each hook item must contain:
  - text
  - family
"""

    secondary_prompt = f"""
Generate 6 stronger, more believable hooks in json for this topic.

Topic: {topic}
Platform: {normalized_platform}
Template: {template_key}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Angle: {angle or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}

Hooks to avoid:
{json.dumps(banned_texts[:20], ensure_ascii=False)}
"""

    def normalize_fn(raw):
        hooks_raw = raw.get("hooks", []) if isinstance(raw, dict) else []
        return _normalize_hook_items(
            hooks_raw,
            normalized_platform=normalized_platform,
            banned_texts=banned_texts,
            template_key=template_key,
            support_first_person=support_first_person,
        )

    def fallback_fn():
        fallback = _fallback_hooks(topic, normalized_platform, template_key)
        filtered: list[dict[str, Any]] = []
        for item in fallback:
            text = item["text"]
            if any(_similarity(text, banned) >= 0.72 for banned in banned_texts):
                continue
            if any(_similarity(text, existing["text"]) >= 0.72 for existing in filtered):
                continue
            filtered.append(item)
        filtered.sort(key=lambda item: item["score"], reverse=True)
        return filtered[:6] if filtered else fallback[:6]

    return _generate_with_retry(
        tool_name="generate_hooks",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=4,
        min_secondary=3,
        fallback_fn=fallback_fn,
        max_output_tokens=1400,
    )


def generate_single_hook(
    topic: str,
    platform: str | None = None,
    template: str | None = "curiosity",
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
    avoid_hooks: Optional[List[str]] = None,
) -> dict[str, Any]:
    hooks = generate_hooks(
        topic=topic,
        platform=platform,
        template=template,
        language=language,
        audience=audience,
        goal=goal,
        offer=offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        avoid_phrases=avoid_phrases,
        extra_context=extra_context,
        avoid_hooks=avoid_hooks,
    )
    return hooks[0]


def generate_titles(
    topic: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    angle: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    support_first_person = _supports_first_person_context(
        extra_context=extra_context,
        angle=angle,
        brand_voice=brand_voice,
    )

    context = {
        "topic": topic,
        "platform": "YouTube",
        "audience": audience,
        "goal": goal,
        "angle": angle,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
    }

    first_person_rule = (
        "First-person titles are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person personal-story claims."
    )

    primary_instructions = f"""
You create high-performing YouTube titles in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 6 title options in a JSON object under "titles".
- Make each title meaningfully different in structure, not just word swaps.
- Spread across these archetypes: {", ".join(TITLE_ARCHETYPES)}.
- Keep titles natural, specific, and varied.
- Avoid repetitive openers and repetitive syntax.
- Use keywords naturally, not awkwardly.
- Do not use fake urgency or cheap clickbait.
- Do not invent proof, revenue, freedom, or personal transformation claims.
- {first_person_rule}
- Prefer specific tradeoffs, mistakes, assumptions, or frameworks over broad motivation.
- Each item should be an object with:
  - text
  - archetype
  - keyword_focus
"""

    primary_prompt = f"""
Create 6 distinct YouTube titles in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You write stronger YouTube titles in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 6 titles under "titles".
- Prioritize clarity, curiosity, realism, and relevance.
- Make them feel specific to the audience and goal.
- Avoid repetitive title shells, fake case-study claims, broad motivational framing, and generic marketing hype.
- {first_person_rule}
- Each item should contain:
  - text
  - archetype
  - keyword_focus
"""

    secondary_prompt = f"""
Generate 6 stronger YouTube titles in json.

Topic: {topic}
Audience: {audience or ""}
Goal: {goal or ""}
Angle: {angle or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_title_items(
            raw.get("titles", []) if isinstance(raw, dict) else [],
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="generate_titles",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=4,
        min_secondary=3,
        fallback_fn=lambda: _fallback_titles(topic),
        max_output_tokens=1200,
    )


def generate_captions(
    topic: str,
    platform: str,
    tone: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    support_first_person = _supports_first_person_context(
        offer=offer,
        extra_context=extra_context,
        angle=angle,
        brand_voice=brand_voice,
    )

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "angle": angle,
        "call_to_action": call_to_action,
        "tone": tone,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
    }

    first_person_rule = (
        "First-person captions are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person success-story or confession framing."
    )

    primary_instructions = f"""
You write platform-native social captions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 caption options under "captions".
- Each caption must feel meaningfully different in voice and structure.
- Use these style buckets once each when possible: {", ".join(CAPTION_STYLES)}.
- Adapt to the user's platform, goal, and audience.
- Use a CTA naturally. Avoid robotic wording.
- Avoid repetitive first sentences and generic "this changes everything" phrasing.
- Do not invent personal wins, audience size, revenue, or transformation claims.
- {first_person_rule}
- Avoid generic community-copy like "join a growing community" unless the user explicitly gave community context.
- Avoid broad motivational filler when practical specificity would work better.
- Each item should be an object with:
  - text
  - style
  - cta
"""

    primary_prompt = f"""
Create 4 distinct captions in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You write stronger social captions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 captions under "captions".
- Make each caption distinct in tone, pacing, and opening.
- Prioritize clear value, audience relevance, and stronger CTA flow.
- Avoid fake testimonials, fake stories, generic community prompts, and hype-heavy language.
- {first_person_rule}
- Prefer specific actions, named habits, tradeoffs, or frameworks over broad motivation.
- Each item must contain:
  - text
  - style
  - cta
"""

    secondary_prompt = f"""
Generate 4 stronger captions in json.

Topic: {topic}
Platform: {normalized_platform}
Tone: {tone}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Angle: {angle or ""}
CTA: {call_to_action or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_caption_items(
            raw.get("captions", []) if isinstance(raw, dict) else [],
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="generate_captions",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=3,
        fallback_fn=lambda: _fallback_captions(topic, normalized_platform, tone),
        max_output_tokens=1500,
    )


def generate_hashtags(
    topic: str,
    platform: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    content_type: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "content_type": content_type,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You create strategic hashtag sets in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 hashtag sets under "sets".
- Each set should have:
  - title
  - description
  - tags
- tags must be an array of hashtags.
- Make the sets meaningfully different:
  1) broad discovery
  2) balanced relevance
  3) niche intent
- Avoid repeated tags across all sets as much as possible.
- Do not include plain words without #.
"""

    primary_prompt = f"""
Create 3 distinct hashtag sets in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger hashtag strategies in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 sets under "sets".
- Each set must feel strategically different, not just shuffled tags.
- Keep tags relevant to topic, audience, goal, and platform.
- Each set must include:
  - title
  - description
  - tags
"""

    secondary_prompt = f"""
Generate 3 stronger hashtag sets in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Content type: {content_type or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_hashtag_sets(raw.get("sets", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_hashtags",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_hashtags(topic, normalized_platform),
        max_output_tokens=1200,
    )


def generate_description_rewrites(
    topic: str,
    platform: str,
    original_description: str = "",
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    support_first_person = _supports_first_person_context(
        offer=offer,
        extra_context=extra_context,
        angle=angle,
        brand_voice=brand_voice,
    )

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "angle": angle,
        "call_to_action": call_to_action,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
        "original_description": _clean_multiline(original_description),
    }

    first_person_rule = (
        "First-person rewrites are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person story or testimonial framing."
    )

    primary_instructions = f"""
You rewrite descriptions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 rewrites under "rewrites".
- Each rewrite must be substantially different in positioning.
- Use these approaches:
  1) SEO-focused
  2) audience-focused
  3) CTA-focused
- Each item should contain:
  - title
  - text
- Keep wording natural and specific.
- If an original description exists, improve it rather than lightly paraphrasing it.
- Do not invent outcomes, fake social proof, or personal transformation claims.
- {first_person_rule}
- Avoid broad motivation, generic community lines, and vague marketing filler.
"""

    primary_prompt = f"""
Create 3 description rewrites in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You write stronger platform descriptions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 rewrites under "rewrites".
- Make each rewrite feel clearly different in positioning and value delivery.
- Keep them more specific and less generic.
- Avoid hype-heavy marketing language, unsupported story claims, and generic community prompts.
- {first_person_rule}
- Prefer named steps, frameworks, mistakes, or practical payoffs over broad inspiration.
- Each item should contain:
  - title
  - text
"""

    secondary_prompt = f"""
Generate 3 stronger description rewrites in json.

Topic: {topic}
Platform: {normalized_platform}
Original description: {_clean_multiline(original_description)}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Angle: {angle or ""}
CTA: {call_to_action or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_description_items(
            raw.get("rewrites", []) if isinstance(raw, dict) else [],
            normalized_platform,
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="generate_description_rewrites",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_descriptions(topic, normalized_platform),
        max_output_tokens=1600,
    )


def generate_scripts(
    topic: str,
    platform: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: str | None = None,
    brand_voice: str | None = None,
    style: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    support_first_person = _supports_first_person_context(
        offer=offer,
        extra_context=extra_context,
        angle=angle,
        brand_voice=brand_voice,
    )

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "angle": angle,
        "call_to_action": call_to_action,
        "keywords": _parse_csvish(keywords),
        "brand_voice": brand_voice,
        "extra_context": extra_context,
        "content_type": style,
    }

    first_person_rule = (
        "First-person scripts are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person success-story or confession framing."
    )

    primary_instructions = f"""
You write short-form creator scripts in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 scripts under "scripts".
- Make each script structurally different.
- Use these styles when possible: {", ".join(SCRIPT_STYLES)}.
- Each item should contain:
  - title
  - style
  - hook
  - body
  - cta
- Body should be concise but complete.
- Do not make all scripts sound like the same template with one word changed.
- Avoid generic filler and unsupported hype.
- Do not invent personal wins, freedom timelines, revenue jumps, or fake case-study framing.
- {first_person_rule}
- Avoid broad motivational intros when named steps, tradeoffs, or habits would be stronger.
"""

    primary_prompt = f"""
Create 3 distinct short-form scripts in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You write stronger short-form scripts in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 scripts under "scripts".
- Each script must feel distinct in structure, pacing, and angle.
- Make the hook, body, and CTA work together naturally.
- Avoid fake guru energy, inflated claims, unsupported personal-story wins, and generic community phrasing.
- {first_person_rule}
- Prefer named actions, frameworks, or mistakes over broad inspiration.
- Each item should contain:
  - title
  - style
  - hook
  - body
  - cta
"""

    secondary_prompt = f"""
Generate 3 stronger short-form scripts in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Angle: {angle or ""}
CTA: {call_to_action or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Style: {style or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_script_items(
            raw.get("scripts", []) if isinstance(raw, dict) else [],
            normalized_platform,
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="generate_scripts",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_scripts(topic, normalized_platform),
        max_output_tokens=1800,
    )


def generate_humanized_rewrites(
    original_text: str,
    *,
    tone: str = "Natural",
    platform: str = "General",
    language: str = "en",
    audience: str | None = None,
    humanization_strength: str = "Balanced",
    preserve_original_meaning: bool = True,
    style_notes: str | None = None,
) -> list[dict[str, Any]]:
    original_clean = _clean_multiline(original_text)
    if not original_clean:
        raise HTTPException(status_code=400, detail="Original text is required.")

    normalized_platform = _normalize_humanizer_platform(platform)
    normalized_tone = _normalize_humanizer_tone(tone)
    normalized_strength = _normalize_humanizer_strength(humanization_strength)

    context = {
        "original_text": original_clean,
        "platform": normalized_platform,
        "tone": normalized_tone,
        "audience": audience,
        "humanization_strength": normalized_strength,
        "preserve_original_meaning": preserve_original_meaning,
        "style_notes": style_notes,
    }

    meaning_rule = (
        "Keep the original meaning, promise, and key facts intact."
        if preserve_original_meaning
        else "You may reshape the wording more aggressively, but do not invent facts or change core intent."
    )

    strength_rule = {
        "Light": "Make subtle improvements only. Keep structure fairly close to the original.",
        "Balanced": "Make the copy noticeably more natural, smoother, and less robotic while keeping the core message intact.",
        "Strong": "Rewrite more boldly to make it feel clearly human, original, and confident, while preserving the core message and factual meaning.",
    }.get(normalized_strength, "Make the copy more natural while keeping the core message intact.")

    primary_instructions = f"""
You humanize AI-written copy in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrite options under "rewrites".
- The first should be the strongest overall version.
- Each rewrite must feel genuinely human, natural, and creator-ready.
- Remove robotic phrasing, filler, repetition, stiff transitions, and obvious AI wording.
- Adapt the wording for the selected platform and tone.
- {meaning_rule}
- {strength_rule}
- Do not invent proof, testimonials, personal experiences, numbers, or outcomes not present in the original text.
- Do not add hype-heavy phrases, fake urgency, or clickbait.
- Do not make all versions small variations of the same sentence pattern.
- Keep the writing clear, personal, and believable.
- Each rewrite item must contain:
  - version_name
  - text
  - why
"""

    primary_prompt = f"""
Create 4 humanized rewrites in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You rewrite copy so it sounds more human in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrites under "rewrites".
- Make each version distinct in rhythm, phrasing, and flow.
- Reduce AI-sounding words, generic marketing lines, and over-polished filler.
- Keep the output natural for {normalized_platform} and in a {normalized_tone.lower()} tone.
- {meaning_rule}
- Do not invent details or drift away from the original intent.
- Each rewrite item must contain:
  - version_name
  - text
  - why
"""

    secondary_prompt = f"""
Generate 4 stronger humanized rewrites in json.

Original text:
{original_clean}

Platform: {normalized_platform}
Tone: {normalized_tone}
Audience: {audience or ""}
Humanization strength: {normalized_strength}
Preserve original meaning: {"Yes" if preserve_original_meaning else "No"}
Style notes: {style_notes or ""}
"""

    def normalize_fn(raw):
        return _normalize_humanizer_items(
            raw.get("rewrites", []) if isinstance(raw, dict) else [],
            original_text=original_clean,
            preserve_original_meaning=preserve_original_meaning,
        )

    return _generate_with_retry(
        tool_name="generate_humanized_rewrites",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_humanizer_rewrites(
            original_clean,
            normalized_tone,
            normalized_platform,
            preserve_original_meaning,
        ),
        max_output_tokens=2200,
    )


def analyze_thumbnail_text(
    topic: str,
    thumbnail_text: str,
    *,
    language: str = "en",
    audience: str | None = None,
    angle: str | None = None,
    desired_outcome: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    support_first_person = _supports_first_person_context(
        extra_context=extra_context,
        angle=angle,
    )

    context = {
        "topic": topic,
        "platform": "YouTube",
        "audience": audience,
        "angle": angle,
        "desired_outcome": desired_outcome,
        "thumbnail_text": thumbnail_text,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
    }

    first_person_rule = (
        "First-person thumbnail suggestions are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person wording in thumbnail suggestions."
    )

    primary_instructions = f"""
You analyze and improve thumbnail text in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return:
  - analysis: object with text, feedback, metrics
  - suggestions: array of 5 improved thumbnail text options
- Thumbnail text should stay short, clear, and clickable.
- Avoid full-sentence clutter.
- Suggestions must be distinct, not minor rewrites of each other.
- Do not invent exaggerated claims, fake transformations, or unsupported proof.
- {first_person_rule}
- Prefer concise, specific tension over broad marketing phrasing.
"""

    primary_prompt = f"""
Analyze thumbnail text and suggest better options in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You improve thumbnail text in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return an object with:
  - analysis
  - suggestions
- Keep suggestions short, punchy, and more clickable.
- Avoid repetitive suggestions, hype-heavy wording, and fake proof language.
- {first_person_rule}
- Prefer concrete contrast, named mistakes, or clearer payoff over generic motivation.
"""

    secondary_prompt = f"""
Generate stronger thumbnail analysis and suggestions in json.

Topic: {topic}
Thumbnail text: {thumbnail_text}
Audience: {audience or ""}
Angle: {angle or ""}
Desired outcome: {desired_outcome or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_thumbnail_result(
            raw,
            topic,
            thumbnail_text,
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="analyze_thumbnail_text",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=1,
        min_secondary=1,
        fallback_fn=lambda: _fallback_thumbnail(topic, thumbnail_text),
        max_output_tokens=1200,
    )