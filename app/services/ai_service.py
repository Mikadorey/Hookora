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
    "website": "Website",
    "landing page": "Landing Page",
    "sales page": "Sales Page",
    "email": "Email",
    "general": "General",
    "whatsapp": "WhatsApp",
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
    "facebook": "Facebook",
    "website": "Website",
    "email": "Email",
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

HOOK_ANGLE_TYPES = [
    "contrarian",
    "mistake",
    "hidden_truth",
    "pain_point",
    "identity",
    "urgency",
    "proof",
    "curiosity",
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

CTA_STYLES = [
    "direct_action",
    "comment_prompt",
    "save_share",
    "soft_conversion",
]

CAROUSEL_STYLES = [
    "educational_breakdown",
    "mistake_series",
    "framework_series",
]

HOOK_SCORE_DIMENSIONS = [
    "clarity",
    "curiosity",
    "specificity",
    "believability",
    "platform_fit",
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

    if platform in {"x", "linkedin", "facebook", "instagram", "whatsapp"}:
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
    add("Source text", context.get("source_text"))
    add("Comment text", context.get("comment_text"))
    add("Trend input", context.get("trend_input"))
    add("Brand samples", context.get("brand_samples"))

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


def _score_hook_angle_item(angle_title: str, sample_hook: str, explanation: str) -> int:
    score = 58

    angle_title_clean = _clean_text(angle_title)
    sample_hook_clean = _clean_text(sample_hook)
    explanation_clean = _clean_multiline(explanation)

    if angle_title_clean:
        score += 6
    if sample_hook_clean:
        score += 10
    if explanation_clean:
        score += 6

    if len(sample_hook_clean.split()) >= 5:
        score += 4

    if not _has_unrealistic_claims(sample_hook_clean):
        score += 5
    else:
        score -= 10

    if not _has_generic_marketing(sample_hook_clean):
        score += 4
    else:
        score -= 8

    if not _has_ai_tell(sample_hook_clean):
        score += 5
    else:
        score -= 8

    if re.search(r"\b(why|mistake|truth|reason|hidden|wrong|stop|most people)\b", sample_hook_clean, re.IGNORECASE):
        score += 8

    return _clamp(score, 40, 98)


def _score_cta_text(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_multiline(text)
    score = 56

    if 8 <= len(cleaned.split()) <= 30:
        score += 12
    elif 5 <= len(cleaned.split()) <= 40:
        score += 6

    if re.search(r"\b(comment|save|share|follow|subscribe|download|join|reply|dm|message|click|start)\b", cleaned, re.IGNORECASE):
        score += 14

    if re.search(r"\b(you|your)\b", cleaned, re.IGNORECASE):
        score += 8

    if not _has_generic_marketing(cleaned):
        score += 6
    else:
        score -= 8

    if not _has_unrealistic_claims(cleaned):
        score += 4
    else:
        score -= 10

    if _has_broad_motivation(cleaned):
        score -= 6

    if not support_first_person and any(re.search(pattern, cleaned.lower()) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 40, 98)


def _score_repurpose_item(text: str) -> int:
    cleaned = _clean_multiline(text)
    score = 58

    if len(cleaned) >= 80:
        score += 10
    if "\n" in cleaned:
        score += 6
    if re.search(r"\b(you|your|why|how|mistake|save|comment|share|start)\b", cleaned, re.IGNORECASE):
        score += 10
    if not _has_generic_marketing(cleaned):
        score += 6
    else:
        score -= 8
    if not _has_unrealistic_claims(cleaned):
        score += 6
    else:
        score -= 10
    if not _has_ai_tell(cleaned):
        score += 6
    else:
        score -= 8

    return _clamp(score, 40, 98)


def _score_ad_copy_item(primary_text: str, headline: str, cta: str) -> int:
    text = f"{primary_text} {headline} {cta}".strip()
    score = 58

    if 40 <= len(primary_text) <= 260:
        score += 12
    if 4 <= len(_clean_text(headline).split()) <= 12:
        score += 10
    if _clean_text(cta):
        score += 8
    if re.search(r"\b(you|your|why|stop|better|start|save|grow|faster)\b", text, re.IGNORECASE):
        score += 10
    if not _has_generic_marketing(text):
        score += 6
    else:
        score -= 8
    if not _has_unrealistic_claims(text):
        score += 6
    else:
        score -= 10

    return _clamp(score, 40, 98)


def _score_carousel_item(hook_slide: str, slides: list[str], closing_slide: str) -> int:
    joined = " ".join([hook_slide] + slides + [closing_slide]).strip()
    score = 60

    if _clean_text(hook_slide):
        score += 10
    if len(slides) >= 3:
        score += 10
    if _clean_text(closing_slide):
        score += 8
    if re.search(r"\b(why|mistake|framework|step|reason|better|stop|most people)\b", joined, re.IGNORECASE):
        score += 10
    if not _has_generic_marketing(joined):
        score += 6
    else:
        score -= 8
    if not _has_unrealistic_claims(joined):
        score += 6
    else:
        score -= 10

    return _clamp(score, 40, 98)


def _score_positioning_item(positioning_angle: str, value_proposition: str, differentiator: str) -> int:
    joined = f"{positioning_angle} {value_proposition} {differentiator}".strip()
    score = 60

    if _clean_text(positioning_angle):
        score += 8
    if _clean_text(value_proposition):
        score += 10
    if _clean_text(differentiator):
        score += 8
    if re.search(r"\b(for|without|faster|clearer|better|simpler|instead of)\b", joined, re.IGNORECASE):
        score += 8
    if not _has_generic_marketing(joined):
        score += 6
    else:
        score -= 8
    if not _has_unrealistic_claims(joined):
        score += 6
    else:
        score -= 10

    return _clamp(score, 40, 98)


def _score_clarity_dimension(text: str) -> int:
    cleaned = _clean_text(text)
    words = cleaned.split()
    score = 56

    if 5 <= len(words) <= 14:
        score += 12
    elif 4 <= len(words) <= 18:
        score += 6
    else:
        score -= 4

    if len(cleaned) <= 90:
        score += 8
    elif len(cleaned) <= 120:
        score += 4
    else:
        score -= 6

    if re.search(r"\b(why|how|mistake|reason|truth|cost|signal|framework|problem)\b", cleaned, re.IGNORECASE):
        score += 10

    if re.search(r"\b(this|that|it)\b", cleaned, re.IGNORECASE) and not re.search(
        r"\b(why|how|mistake|reason|truth|cost|signal|framework|problem)\b",
        cleaned,
        re.IGNORECASE,
    ):
        score -= 4

    if any(re.search(pattern, cleaned.lower()) for pattern in GENERIC_HOOK_PATTERNS):
        score -= 8

    return _clamp(score, 35, 98)


def _score_curiosity_dimension(text: str) -> int:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    score = 54

    if re.search(r"\b(why|what|how|mistake|truth|reason|hidden|wrong|miss|cost)\b", lowered):
        score += 14

    if re.search(r"\b(most people|nobody|quietly|really|actually|still)\b", lowered):
        score += 8

    if re.search(r"\b(secret|insane|crazy|overnight|guaranteed)\b", lowered):
        score -= 10

    if any(re.search(pattern, lowered) for pattern in WEAK_WARNING_PATTERNS):
        score -= 6

    return _clamp(score, 35, 98)


def _score_specificity_dimension(text: str) -> int:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    score = 52

    if re.search(r"\b(framework|mistake|tradeoff|assumption|timeline|cost|pattern|signal|question|rule)\b", lowered):
        score += 16

    if re.search(r"\b(your|this|that)\b", lowered):
        score += 4

    if any(re.search(pattern, lowered) for pattern in GENERIC_HOOK_PATTERNS):
        score -= 10

    if re.search(r"\b(things|stuff|everything|anything)\b", lowered):
        score -= 8

    return _clamp(score, 35, 98)


def _score_believability_dimension(text: str, support_first_person: bool = False) -> int:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    score = 58

    if any(word in lowered for word in BELIEVABILITY_WORDS):
        score += 12

    if _has_unrealistic_claims(cleaned):
        score -= 16

    if re.search(r"\b(secret|guaranteed|instantly|overnight|always works|never fails)\b", lowered):
        score -= 12

    if not support_first_person and any(re.search(pattern, lowered) for pattern in FIRST_PERSON_PATTERNS):
        score -= 10

    return _clamp(score, 35, 98)


def _score_platform_fit_dimension(text: str, platform: str = "General") -> int:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    normalized_platform = _normalize_platform(platform).lower()
    score = 54

    words = len(cleaned.split())

    if normalized_platform == "youtube":
        if 8 <= words <= 15:
            score += 14
        elif 6 <= words <= 17:
            score += 8
        else:
            score -= 4

        if any(word in lowered for word in YOUTUBE_HOOK_WORDS):
            score += 8

    elif normalized_platform in {"youtube shorts", "instagram reels", "tiktok"}:
        if 5 <= words <= 11:
            score += 14
        elif 4 <= words <= 13:
            score += 8
        else:
            score -= 4

        if any(word in lowered for word in SHORTFORM_HOOK_WORDS):
            score += 8

    else:
        if 6 <= words <= 14:
            score += 12
        elif 5 <= words <= 16:
            score += 6
        else:
            score -= 4

    return _clamp(score, 35, 98)
def _hook_score_feedback(
    *,
    clarity: int,
    curiosity: int,
    specificity: int,
    believability: int,
    platform_fit: int,
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    if clarity >= 75:
        strengths.append("Clear and easy to understand quickly.")
    elif clarity < 60:
        weaknesses.append("The wording is not clear enough on first read.")

    if curiosity >= 75:
        strengths.append("Creates a strong reason to keep reading or watching.")
    elif curiosity < 60:
        weaknesses.append("It needs a stronger curiosity gap or tension point.")

    if specificity >= 75:
        strengths.append("Feels specific instead of generic.")
    elif specificity < 60:
        weaknesses.append("It sounds too broad and could be more concrete.")

    if believability >= 75:
        strengths.append("Feels credible and grounded.")
    elif believability < 60:
        weaknesses.append("It risks sounding exaggerated or less believable.")

    if platform_fit >= 75:
        strengths.append("Fits the selected platform well.")
    elif platform_fit < 60:
        weaknesses.append("It does not fully match the rhythm or style of the platform.")

    if not strengths:
        strengths.append("There is a workable base idea to build on.")

    if not weaknesses:
        weaknesses.append("The hook is solid overall, but can still be sharpened.")

    return strengths[:3], weaknesses[:3]


def _hook_score_verdict(score: int) -> str:
    if score >= 85:
        return "Excellent hook foundation."
    if score >= 70:
        return "Strong hook with room to sharpen."
    if score >= 55:
        return "Decent hook, but it needs stronger tension or clarity."
    return "Weak hook. It needs a clearer, more specific, and more compelling angle."


def _fallback_hook_score_analysis(
    hook_text: str,
    *,
    platform: str = "General",
    support_first_person: bool = False,
) -> dict[str, Any]:
    cleaned = _postprocess_hook_text(hook_text)
    if not cleaned:
        raise ValueError("Hook text is required.")

    clarity = _score_clarity_dimension(cleaned)
    curiosity = _score_curiosity_dimension(cleaned)
    specificity = _score_specificity_dimension(cleaned)
    believability = _score_believability_dimension(
        cleaned,
        support_first_person=support_first_person,
    )
    platform_fit = _score_platform_fit_dimension(cleaned, platform=platform)

    overall_score = round(
        (
            clarity * 0.24
            + curiosity * 0.24
            + specificity * 0.18
            + believability * 0.20
            + platform_fit * 0.14
        )
    )

    strengths, weaknesses = _hook_score_feedback(
        clarity=clarity,
        curiosity=curiosity,
        specificity=specificity,
        believability=believability,
        platform_fit=platform_fit,
    )

    improved_hook = cleaned
    if overall_score < 70 and not re.search(
        r"\b(why|how|mistake|truth|reason|cost|signal|framework)\b",
        cleaned,
        re.IGNORECASE,
    ):
        improved_hook = f"What most people miss about {cleaned.lower()}"

    return {
        "hook_text": cleaned,
        "overall_score": _clamp(overall_score, 35, 98),
        "label": _title_label(_clamp(overall_score, 35, 98)),
        "verdict": _hook_score_verdict(_clamp(overall_score, 35, 98)),
        "dimensions": {
            "clarity": clarity,
            "curiosity": curiosity,
            "specificity": specificity,
            "believability": believability,
            "platform_fit": platform_fit,
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improved_hook": improved_hook,
    }


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
        {
            "text": item,
            "score": _score_title_text(item),
            "label": _title_label(_score_title_text(item)),
        }
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
        {
            "text": item,
            "score": _score_caption_text(item),
            "label": _title_label(_score_caption_text(item)),
        }
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
        result.append(
            {
                **item,
                "score": score,
                "label": _title_label(score),
                "platform": platform,
            }
        )
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
        normalized.append(
            {
                **item,
                "text": text,
                "score": score,
                "label": _title_label(score),
                "platform": platform,
            }
        )
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


def _fallback_hook_angles(topic: str, platform: str) -> list[dict[str, Any]]:
    topic = _clean_text(topic)
    platform_label = _normalize_platform(platform)

    items = [
        {
            "angle_type": "contrarian",
            "angle_title": f"Most people are framing {topic} the wrong way",
            "explanation": f"Use a contrarian angle to challenge common advice and make people stop scrolling on {platform_label}.",
            "sample_hook": f"Most people are overcomplicating {topic}.",
            "best_use_case": "Works well when your audience has seen the same recycled advice too many times.",
        },
        {
            "angle_type": "mistake",
            "angle_title": f"The common mistake hurting {topic}",
            "explanation": "Lead with a mistake your audience is likely making so the message feels immediately relevant.",
            "sample_hook": f"This is the mistake making your {topic} weaker than it should be.",
            "best_use_case": "Best for educational, marketing, and creator content.",
        },
        {
            "angle_type": "hidden_truth",
            "angle_title": f"What people miss about {topic}",
            "explanation": "Reveal a hidden truth or overlooked detail to create curiosity without sounding fake.",
            "sample_hook": f"What people miss about {topic} is usually the part that matters most.",
            "best_use_case": "Great when you want curiosity with a more thoughtful tone.",
        },
        {
            "angle_type": "pain_point",
            "angle_title": f"Why {topic} still feels frustrating",
            "explanation": "Speak directly to the frustration or bottleneck the audience is feeling right now.",
            "sample_hook": f"If {topic} still feels frustrating, this is probably why.",
            "best_use_case": "Best when the audience is actively struggling with a problem.",
        },
        {
            "angle_type": "urgency",
            "angle_title": f"The cost of ignoring {topic}",
            "explanation": "Show what gets worse when the audience delays action.",
            "sample_hook": f"Ignoring this part of {topic} quietly costs you more than you think.",
            "best_use_case": "Useful for action-driven content, sales content, and business education.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_hook_angle_item(
            item["angle_title"],
            item["sample_hook"],
            item["explanation"],
        )
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    if normalized:
        normalized[0]["angle_title"] = f"Best Angle: {normalized[0]['angle_title']}"
    return normalized[:5]
def _fallback_ctas(topic: str, platform: str) -> list[dict[str, Any]]:
    platform_label = _normalize_platform(platform)
    topic = _clean_text(topic)

    items = [
        {
            "text": f"If this helped you think differently about {topic}, save it and follow for more.",
            "style": "direct_action",
            "why": f"Clear follow-and-save CTA that fits {platform_label}.",
        },
        {
            "text": f"What’s your biggest challenge with {topic}? Drop it in the comments.",
            "style": "comment_prompt",
            "why": "Invites engagement through a simple question.",
        },
        {
            "text": f"Save this for later and share it with someone working on {topic}.",
            "style": "save_share",
            "why": "Encourages both saves and shares without sounding pushy.",
        },
        {
            "text": f"If you want better results with {topic}, follow for more practical breakdowns.",
            "style": "soft_conversion",
            "why": "Soft conversion CTA focused on future value.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_cta_text(item["text"])
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:4]


def _fallback_repurpose_outputs(topic: str, platform: str, source_text: str) -> list[dict[str, Any]]:
    short_source = _clean_text(source_text)[:180]
    topic = _clean_text(topic or "your idea")
    platform = _normalize_platform(platform)

    items = [
        {
            "format_name": "Hook-Led Social Post",
            "text": f"Most people think {topic} is about doing more.\n\nIt usually is not.\n\nThe real shift is making the message sharper, clearer, and easier to act on.\n\n{short_source}\n\nSave this if you want stronger content on {platform}.",
            "why": "Starts with tension, then turns the original idea into a cleaner social post.",
        },
        {
            "format_name": "Short Caption Version",
            "text": f"If your {topic} still feels flat, the issue may be positioning, not effort.\n\n{short_source}\n\nComment if you want a stronger version of this idea.",
            "why": "Tighter version for shorter-feed formats and quick engagement.",
        },
        {
            "format_name": "Value-Driven Rewrite",
            "text": f"Here’s the smarter way to think about {topic}:\n\n{short_source}\n\nWhat matters most is clarity, relevance, and strong framing.\n\nShare this with someone building content right now.",
            "why": "Keeps the core message but makes the takeaway easier to remember and share.",
        },
        {
            "format_name": "Thread / Carousel Starter",
            "text": f"Most people miss this about {topic}.\n\n1. Weak messaging usually sounds fine to the creator.\n2. The audience only notices what feels relevant fast.\n3. Sharper framing changes everything.\n\n{short_source}\n\nWant more breakdowns like this? Follow along.",
            "why": "Turns one idea into a more structured multi-part format.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item(item["text"])
        normalized.append({**item, "score": score, "label": _title_label(score)})
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:4]


def _fallback_ad_copy(topic: str, platform: str, offer: str | None = None) -> list[dict[str, Any]]:
    topic = _clean_text(topic or "your offer")
    platform = _normalize_platform(platform)
    offer_clean = _clean_text(offer or topic)

    items = [
        {
            "variant_name": "Pain Point Variant",
            "primary_text": f"Still spending too much time trying to make {topic} work? {offer_clean} helps you move faster with clearer messaging and stronger content decisions.",
            "headline": f"Make {topic} easier",
            "description": f"Built for people who want stronger results without wasting time.",
            "cta": "Start Now",
        },
        {
            "variant_name": "Outcome Variant",
            "primary_text": f"If you want better results from {topic}, start with a clearer system. {offer_clean} helps you create stronger content with less guesswork.",
            "headline": f"Get better {topic} results",
            "description": f"A smarter way to improve your output on {platform}.",
            "cta": "Try It Free",
        },
        {
            "variant_name": "Differentiator Variant",
            "primary_text": f"Most tools make content faster. {offer_clean} helps make it sharper too. That difference matters when you want stronger engagement and conversion.",
            "headline": "Faster and sharper content",
            "description": f"Built for clarity, speed, and better message positioning.",
            "cta": "See How",
        },
    ]

    normalized = []
    for item in items:
        score = _score_ad_copy_item(item["primary_text"], item["headline"], item["cta"])
        normalized.append({**item, "score": score, "label": _title_label(score)})
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_carousels(topic: str, platform: str, style: str | None = None) -> list[dict[str, Any]]:
    topic = _clean_text(topic or "your topic")
    platform = _normalize_platform(platform)
    style_clean = _clean_text(style or "educational_breakdown")

    items = [
        {
            "title": f"{topic}: What Most People Get Wrong",
            "style": style_clean,
            "hook_slide": f"What most people get wrong about {topic}",
            "slides": [
                f"Most people focus on output before clarity in {topic}.",
                f"The message usually gets weaker when the angle is too broad.",
                f"The better move is sharper positioning and clearer payoff.",
            ],
            "closing_slide": f"Save this if you want stronger {topic} results.",
            "caption": f"If your {topic} content feels average, start by fixing the framing before doing more work.",
        },
        {
            "title": f"3 Mistakes Hurting {topic}",
            "style": "mistake_series",
            "hook_slide": f"3 mistakes hurting your {topic}",
            "slides": [
                "Trying to say too many things at once.",
                "Leading with information instead of tension.",
                "Using vague copy that sounds fine but feels forgettable.",
            ],
            "closing_slide": "Comment if you want the fix for each one.",
            "caption": f"Most weak {topic} content is not missing effort. It is missing stronger structure and sharper positioning.",
        },
        {
            "title": f"A Better Framework for {topic}",
            "style": "framework_series",
            "hook_slide": f"A better framework for {topic}",
            "slides": [
                "Start with the real pain point.",
                "Show the shift or insight clearly.",
                "End with a takeaway people can act on fast.",
            ],
            "closing_slide": f"Follow for more practical {topic} breakdowns.",
            "caption": f"If you want your {topic} to land better, use a framework people can understand and remember quickly.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_carousel_item(item["hook_slide"], item["slides"], item["closing_slide"])
        normalized.append({**item, "score": score, "label": _title_label(score)})
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_viral_rewrites(source_text: str, topic: str | None = None) -> list[dict[str, Any]]:
    base = _clean_multiline(source_text)
    topic_clean = _clean_text(topic or "this")
    if not base:
        raise ValueError("Source text is required.")

    items = [
        {
            "version_name": "Best Rewrite",
            "text": f"Most people are too casual about {topic_clean}.\n\nThat is exactly why their message gets ignored.\n\n{base}",
            "why": "Adds stronger tension and a sharper opening without losing the original idea.",
        },
        {
            "version_name": "Alternative 1",
            "text": f"If your {topic_clean} still feels weak, this may be why:\n\n{base}",
            "why": "Makes the rewrite more direct and problem-led.",
        },
        {
            "version_name": "Alternative 2",
            "text": f"Here is what most people miss about {topic_clean}:\n\n{base}",
            "why": "Uses a curiosity-led opener while staying believable.",
        },
        {
            "version_name": "Alternative 3",
            "text": f"The real problem with most {topic_clean} content is not effort.\nIt is framing.\n\n{base}",
            "why": "Turns the original message into a clearer, more scroll-stopping angle.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item(item["text"])
        normalized.append({**item, "score": score, "label": _title_label(score)})
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:4]


def _fallback_offer_positioning(topic: str, offer: str | None = None) -> list[dict[str, Any]]:
    topic_clean = _clean_text(topic or "your offer")
    offer_clean = _clean_text(offer or topic_clean)

    items = [
        {
            "variant_name": "Clarity-First Positioning",
            "positioning_angle": f"A clearer way to improve {topic_clean}",
            "value_proposition": f"{offer_clean} helps people move faster with clearer decisions and stronger messaging.",
            "who_its_for": "People who want better output without more confusion.",
            "problem_frame": f"Most people struggle with {topic_clean} because the message is not sharp enough.",
            "differentiator": "Focuses on both speed and message quality, not just volume.",
            "message_example": f"{offer_clean} helps you create sharper, more effective content without wasting time.",
        },
        {
            "variant_name": "Outcome-Driven Positioning",
            "positioning_angle": f"Built for better {topic_clean} results",
            "value_proposition": f"{offer_clean} makes it easier to turn weak ideas into stronger content and clearer offers.",
            "who_its_for": "Creators, founders, and marketers who need better performance from their messaging.",
            "problem_frame": "Most solutions make work faster but not better.",
            "differentiator": "Emphasizes stronger positioning, not just convenience.",
            "message_example": f"If you want better results from {topic_clean}, start with a tool that improves the message itself.",
        },
        {
            "variant_name": "Differentiator-Led Positioning",
            "positioning_angle": f"More than a faster {topic_clean} tool",
            "value_proposition": f"{offer_clean} helps people create stronger output that actually feels more relevant, useful, and conversion-ready.",
            "who_its_for": "People tired of generic output and weak framing.",
            "problem_frame": "Fast content still fails when the positioning is weak.",
            "differentiator": "Combines speed with sharper angles, copy, and positioning logic.",
            "message_example": f"{offer_clean} is for people who want faster creation and stronger messaging at the same time.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_positioning_item(
            item["positioning_angle"],
            item["value_proposition"],
            item["differentiator"],
        )
        normalized.append({**item, "score": score, "label": _title_label(score)})
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_retention_intros(topic: str, platform: str) -> list[dict[str, Any]]:
    topic_clean = _clean_text(topic or "your topic")
    platform_clean = _normalize_platform(platform)

    items = [
        {
            "version_name": "Problem-Led Intro",
            "hook_line": f"If your {topic_clean} still is not landing, this is probably why.",
            "preview_line": f"In the next few seconds, I’ll show you what most people miss about {topic_clean}.",
            "lesson_line": f"The real issue usually is not effort. It is weak framing and unclear payoff.",
            "cta_line": f"Stay with this and you’ll leave with a stronger way to approach {topic_clean} on {platform_clean}.",
        },
        {
            "version_name": "Curiosity-Led Intro",
            "hook_line": f"What most people get wrong about {topic_clean} is not obvious at first.",
            "preview_line": f"But once you see it, you start noticing why weak content keeps getting ignored.",
            "lesson_line": f"This breakdown will show you the shift that makes {topic_clean} feel sharper and more watchable.",
            "cta_line": "Watch closely, because this one change affects everything that comes after.",
        },
        {
            "version_name": "Contrarian Intro",
            "hook_line": f"Most advice around {topic_clean} is too shallow.",
            "preview_line": "It sounds useful, but it rarely gives people a reason to care fast.",
            "lesson_line": f"Here is the smarter way to make {topic_clean} feel more relevant and more engaging.",
            "cta_line": "If you want stronger retention, start here.",
        },
    ]

    normalized = []
    for item in items:
        full_text = "\n".join(
            [
                item["hook_line"],
                item["preview_line"],
                item["lesson_line"],
                item["cta_line"],
            ]
        )
        score = _score_script_text(full_text)
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_comment_to_content(comment_text: str, platform: str) -> list[dict[str, Any]]:
    comment_clean = _clean_text(comment_text)
    platform_clean = _normalize_platform(platform)

    items = [
        {
            "content_type": "reply_post",
            "title": "Direct Answer Post",
            "hook": f"Someone asked: {comment_clean}",
            "body": f"That question matters because most people run into it at the exact point where their {platform_clean} content starts feeling weak. The fix usually starts with clarity, not more volume.",
            "cta": "Comment if you want me to break this down further.",
        },
        {
            "content_type": "short_video",
            "title": "Short Video Response",
            "hook": f"Let’s answer this properly: {comment_clean}",
            "body": "There is usually a simpler answer than people expect, but it only makes sense when you look at the positioning, the audience, and the exact friction point.",
            "cta": "Save this if you want more comment-based breakdowns.",
        },
        {
            "content_type": "carousel",
            "title": "Carousel Breakdown",
            "hook": f"This question deserves a full breakdown: {comment_clean}",
            "body": "Instead of replying in one sentence, turn it into a structured lesson people can save, share, and revisit later.",
            "cta": "Follow for more content built from real audience questions.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item("\n".join([item["hook"], item["body"], item["cta"]]))
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]
def _fallback_trend_to_content(trend_input: str, platform: str) -> list[dict[str, Any]]:
    trend_clean = _clean_text(trend_input or "current trend")
    platform_clean = _normalize_platform(platform)

    items = [
        {
            "format_name": "Trend Angle Post",
            "hook": f"Everyone is talking about {trend_clean}, but here’s the part that actually matters.",
            "body": f"Instead of copying the trend directly, turn it into a clearer idea your audience can connect to on {platform_clean}. The strongest version is usually not the loudest one. It is the one with the clearest angle.",
            "cta": "Save this if you want to turn trends into stronger content ideas.",
        },
        {
            "format_name": "Trend Breakdown Script",
            "hook": f"This trend is everywhere right now: {trend_clean}",
            "body": "But most people use it in a shallow way. The smarter move is to connect the trend to a pain point, a lesson, or a strong opinion your audience already cares about.",
            "cta": "Comment if you want more trend-to-content ideas.",
        },
        {
            "format_name": "Trend Twist Carousel",
            "hook": f"How to use {trend_clean} without sounding like everyone else",
            "body": "Start with what people already know, then add a sharper point of view, a useful framework, or a more relevant audience-specific lesson.",
            "cta": "Share this with someone creating content this week.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item("\n".join([item["hook"], item["body"], item["cta"]]))
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_nigerian_rewrites(source_text: str) -> list[dict[str, Any]]:
    base = _clean_multiline(source_text)
    if not base:
        raise ValueError("Source text is required.")

    items = [
        {
            "version_name": "Best Rewrite",
            "text": f"This hits differently when you look at how people actually move in Nigeria.\n\n{base}",
            "why": "Makes the phrasing feel more locally aware without overdoing slang.",
        },
        {
            "version_name": "Alternative 1",
            "text": f"If you dey try make this thing work, here’s the part most people miss:\n\n{base}",
            "why": "Adds a more Nigerian conversational rhythm while keeping it readable.",
        },
        {
            "version_name": "Alternative 2",
            "text": f"To be honest, this is where many people get it wrong:\n\n{base}",
            "why": "Keeps it natural and familiar without sounding forced.",
        },
        {
            "version_name": "Alternative 3",
            "text": f"Most people for here no even realize say this part matters that much.\n\n{base}",
            "why": "Brings in a stronger local feel for a more relatable rewrite.",
        },
    ]

    normalized = []
    seen = []
    for item in items:
        text = _postprocess_humanizer_text(item["text"])
        if not text:
            continue
        if any(_similarity(text, existing) >= 0.86 for existing in seen):
            continue
        seen.append(text)
        score = _score_humanizer_text(text, base, True)
        normalized.append({**item, "text": text, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:4]


def _fallback_brand_voice_outputs(topic: str, brand_samples: str) -> list[dict[str, Any]]:
    topic_clean = _clean_text(topic or "your topic")
    sample_hint = _clean_text(brand_samples)[:180]

    items = [
        {
            "voice_name": "Best Matched Voice",
            "voice_summary": f"A sharper, more confident version of your existing voice, grounded in the way you already communicate about {topic_clean}.",
            "example_output": f"Most people approach {topic_clean} too casually. The better move is clearer thinking, sharper framing, and stronger decisions from the start.",
            "guidance": f"Keep the tone direct, confident, and practical. Avoid overexplaining. Reference style sample: {sample_hint}" if sample_hint else "Keep the tone direct, confident, and practical. Avoid overexplaining.",
        },
        {
            "voice_name": "Alternative Voice 1",
            "voice_summary": f"A warmer, more conversational voice for talking about {topic_clean} without losing authority.",
            "example_output": f"If {topic_clean} has been feeling harder than it should, you may not need more effort. You may just need a clearer angle.",
            "guidance": "Use shorter sentences, cleaner flow, and more audience-focused phrasing.",
        },
        {
            "voice_name": "Alternative Voice 2",
            "voice_summary": f"A bolder voice with stronger tension and stronger conviction around {topic_clean}.",
            "example_output": f"The biggest mistake people make with {topic_clean} is thinking average messaging will still get strong results.",
            "guidance": "Lead with the point faster. Reduce filler. Use stronger contrast and cleaner takeaways.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item(
            "\n".join(
                [
                    item["voice_summary"],
                    item["example_output"],
                    item["guidance"],
                ]
            )
        )
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


def _fallback_content_differentiation(topic: str, audience: str | None = None) -> list[dict[str, Any]]:
    topic_clean = _clean_text(topic or "your niche")
    audience_clean = _clean_text(audience or "your audience")

    items = [
        {
            "variant_name": "Angle Differentiation",
            "differentiator": f"Most people in {topic_clean} repeat the same surface-level advice. Go deeper into mistakes, tradeoffs, and clearer frameworks for {audience_clean}.",
            "opportunity": "Differentiate by saying something more useful, not just something louder.",
            "example": f"Instead of generic tips on {topic_clean}, break down what people quietly get wrong and why it matters.",
        },
        {
            "variant_name": "Voice Differentiation",
            "differentiator": f"Use a more grounded and direct tone than most creators talking about {topic_clean}.",
            "opportunity": "People trust voices that feel clear, useful, and less inflated.",
            "example": f"Talk about {topic_clean} with sharper wording, cleaner structure, and fewer recycled phrases.",
        },
        {
            "variant_name": "Format Differentiation",
            "differentiator": f"Turn {topic_clean} into stronger breakdowns, structured carousels, and response-led content based on real audience friction.",
            "opportunity": "Format can make common ideas feel new again.",
            "example": "Use questions, objections, and comment-driven content to make the message feel more relevant.",
        },
    ]

    normalized = []
    for item in items:
        score = _score_repurpose_item(
            "\n".join(
                [
                    item["differentiator"],
                    item["opportunity"],
                    item["example"],
                ]
            )
        )
        normalized.append({**item, "score": score, "label": _title_label(score)})

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:3]


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
                "score": _score_hook_text(
                    text,
                    normalized_platform,
                    support_first_person=support_first_person,
                ),
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


def _normalize_hook_angle_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen_titles: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        angle_type = _clean_text(item.get("angle_type") or f"angle_{index + 1}")
        angle_title = _clean_text(item.get("angle_title") or "")
        explanation = _clean_multiline(item.get("explanation") or "")
        sample_hook = _postprocess_hook_text(item.get("sample_hook") or "")
        best_use_case = _clean_multiline(item.get("best_use_case") or "")

        if not angle_title or not sample_hook:
            continue

        if any(_similarity(angle_title, existing) >= 0.75 for existing in seen_titles):
            continue

        seen_titles.append(angle_title)
        score = _score_hook_angle_item(angle_title, sample_hook, explanation)
        normalized.append(
            {
                "angle_type": angle_type,
                "angle_title": angle_title,
                "explanation": explanation,
                "sample_hook": sample_hook,
                "best_use_case": best_use_case,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 5:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)

    if normalized:
        normalized[0]["angle_title"] = (
            normalized[0]["angle_title"]
            if normalized[0]["angle_title"].startswith("Best Angle:")
            else f"Best Angle: {normalized[0]['angle_title']}"
        )

    return normalized


def _normalize_cta_items(items_raw: Any, *, support_first_person: bool = False) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if isinstance(item, dict):
            text = _postprocess_general_text(item.get("text") or "")
            style = _clean_text(item.get("style") or "general")
            why = _clean_text(item.get("why") or "")
        else:
            text = _postprocess_general_text(str(item))
            style = "general"
            why = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.74 for existing in seen):
            continue

        seen.append(text)
        score = _score_cta_text(text, support_first_person=support_first_person)
        normalized.append(
            {
                "text": text,
                "style": style or f"cta_{index + 1}",
                "why": why,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 4:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_repurpose_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if isinstance(item, dict):
            format_name = _clean_text(item.get("format_name") or f"Output {index + 1}")
            text = _postprocess_general_text(item.get("text") or "")
            why = _clean_text(item.get("why") or "")
        else:
            format_name = f"Output {index + 1}"
            text = _postprocess_general_text(str(item))
            why = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.76 for existing in seen):
            continue

        seen.append(text)
        score = _score_repurpose_item(text)
        normalized.append(
            {
                "format_name": format_name,
                "text": text,
                "why": why,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 4:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_ad_copy_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        variant_name = _clean_text(item.get("variant_name") or f"Ad Variant {index + 1}")
        primary_text = _postprocess_general_text(item.get("primary_text") or "")
        headline = _clean_text(item.get("headline") or "")
        description = _clean_text(item.get("description") or "")
        cta = _clean_text(item.get("cta") or "")

        joined = " ".join([primary_text, headline, description, cta]).strip()
        if not primary_text or not headline:
            continue
        if any(_similarity(joined, existing) >= 0.78 for existing in seen):
            continue

        seen.append(joined)
        score = _score_ad_copy_item(primary_text, headline, cta)
        normalized.append(
            {
                "variant_name": variant_name,
                "primary_text": primary_text,
                "headline": headline,
                "description": description,
                "cta": cta,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_carousel_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title") or f"Carousel {index + 1}")
        style = _clean_text(item.get("style") or "educational_breakdown")
        hook_slide = _clean_text(item.get("hook_slide") or "")
        slides = item.get("slides") or []
        closing_slide = _clean_text(item.get("closing_slide") or "")
        caption = _postprocess_general_text(item.get("caption") or "")

        if not isinstance(slides, list):
            slides = []

        clean_slides = []
        for slide in slides:
            slide_text = _clean_text(str(slide))
            if slide_text:
                clean_slides.append(slide_text)

        joined = " ".join([hook_slide] + clean_slides + [closing_slide, caption]).strip()
        if not hook_slide or len(clean_slides) < 2:
            continue
        if any(_similarity(joined, existing) >= 0.76 for existing in seen):
            continue

        seen.append(joined)
        score = _score_carousel_item(hook_slide, clean_slides, closing_slide)
        normalized.append(
            {
                "title": title,
                "style": style,
                "hook_slide": hook_slide,
                "slides": clean_slides[:5],
                "closing_slide": closing_slide,
                "caption": caption,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_positioning_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        variant_name = _clean_text(item.get("variant_name") or f"Positioning {index + 1}")
        positioning_angle = _clean_text(item.get("positioning_angle") or "")
        value_proposition = _postprocess_general_text(item.get("value_proposition") or "")
        who_its_for = _clean_text(item.get("who_its_for") or "")
        problem_frame = _postprocess_general_text(item.get("problem_frame") or "")
        differentiator = _postprocess_general_text(item.get("differentiator") or "")
        message_example = _postprocess_general_text(item.get("message_example") or "")

        joined = " ".join(
            [
                variant_name,
                positioning_angle,
                value_proposition,
                who_its_for,
                problem_frame,
                differentiator,
                message_example,
            ]
        ).strip()

        if not positioning_angle or not value_proposition:
            continue
        if any(_similarity(joined, existing) >= 0.78 for existing in seen):
            continue

        seen.append(joined)
        score = _score_positioning_item(positioning_angle, value_proposition, differentiator)
        normalized.append(
            {
                "variant_name": variant_name,
                "positioning_angle": positioning_angle,
                "value_proposition": value_proposition,
                "who_its_for": who_its_for,
                "problem_frame": problem_frame,
                "differentiator": differentiator,
                "message_example": message_example,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized
def _normalize_hook_score_result(
    raw: Any,
    hook_text: str,
    *,
    platform: str = "General",
    support_first_person: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _fallback_hook_score_analysis(
            hook_text,
            platform=platform,
            support_first_person=support_first_person,
        )

    hook_text_clean = _postprocess_hook_text(raw.get("hook_text") or hook_text)
    if not hook_text_clean:
        hook_text_clean = _postprocess_hook_text(hook_text)

    dimensions_raw = raw.get("dimensions", {}) or {}
    clarity = _clamp(int(dimensions_raw.get("clarity", _score_clarity_dimension(hook_text_clean))), 35, 98)
    curiosity = _clamp(int(dimensions_raw.get("curiosity", _score_curiosity_dimension(hook_text_clean))), 35, 98)
    specificity = _clamp(int(dimensions_raw.get("specificity", _score_specificity_dimension(hook_text_clean))), 35, 98)
    believability = _clamp(
        int(
            dimensions_raw.get(
                "believability",
                _score_believability_dimension(
                    hook_text_clean,
                    support_first_person=support_first_person,
                ),
            )
        ),
        35,
        98,
    )
    platform_fit = _clamp(
        int(dimensions_raw.get("platform_fit", _score_platform_fit_dimension(hook_text_clean, platform))),
        35,
        98,
    )

    overall_score = raw.get("overall_score")
    if overall_score is None:
        overall_score = round(
            clarity * 0.24
            + curiosity * 0.24
            + specificity * 0.18
            + believability * 0.20
            + platform_fit * 0.14
        )
    overall_score = _clamp(int(overall_score), 35, 98)

    strengths = raw.get("strengths") or []
    weaknesses = raw.get("weaknesses") or []

    if not isinstance(strengths, list) or not isinstance(weaknesses, list):
        strengths, weaknesses = _hook_score_feedback(
            clarity=clarity,
            curiosity=curiosity,
            specificity=specificity,
            believability=believability,
            platform_fit=platform_fit,
        )

    cleaned_strengths = [str(item).strip() for item in strengths if str(item).strip()][:3]
    cleaned_weaknesses = [str(item).strip() for item in weaknesses if str(item).strip()][:3]

    if not cleaned_strengths or not cleaned_weaknesses:
        fallback_strengths, fallback_weaknesses = _hook_score_feedback(
            clarity=clarity,
            curiosity=curiosity,
            specificity=specificity,
            believability=believability,
            platform_fit=platform_fit,
        )
        if not cleaned_strengths:
            cleaned_strengths = fallback_strengths
        if not cleaned_weaknesses:
            cleaned_weaknesses = fallback_weaknesses

    improved_hook = _postprocess_hook_text(raw.get("improved_hook") or "")
    if not improved_hook:
        improved_hook = _fallback_hook_score_analysis(
            hook_text_clean,
            platform=platform,
            support_first_person=support_first_person,
        )["improved_hook"]

    verdict = _clean_text(raw.get("verdict") or _hook_score_verdict(overall_score))

    return {
        "hook_text": hook_text_clean,
        "overall_score": overall_score,
        "label": _title_label(overall_score),
        "verdict": verdict,
        "dimensions": {
            "clarity": clarity,
            "curiosity": curiosity,
            "specificity": specificity,
            "believability": believability,
            "platform_fit": platform_fit,
        },
        "strengths": cleaned_strengths,
        "weaknesses": cleaned_weaknesses,
        "improved_hook": improved_hook,
    }


def _normalize_retention_intro_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        version_name = _clean_text(item.get("version_name") or f"Intro {index + 1}")
        hook_line = _clean_text(item.get("hook_line") or "")
        preview_line = _clean_text(item.get("preview_line") or "")
        lesson_line = _clean_text(item.get("lesson_line") or "")
        cta_line = _clean_text(item.get("cta_line") or "")

        joined = "\n".join([hook_line, preview_line, lesson_line, cta_line]).strip()
        if not hook_line or not lesson_line:
            continue
        if any(_similarity(joined, existing) >= 0.76 for existing in seen):
            continue

        seen.append(joined)
        score = _score_script_text(joined)
        normalized.append(
            {
                "version_name": version_name,
                "hook_line": hook_line,
                "preview_line": preview_line,
                "lesson_line": lesson_line,
                "cta_line": cta_line,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_comment_to_content_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        content_type = _clean_text(item.get("content_type") or f"format_{index + 1}")
        title = _clean_text(item.get("title") or f"Idea {index + 1}")
        hook = _clean_text(item.get("hook") or "")
        body = _postprocess_general_text(item.get("body") or "")
        cta = _clean_text(item.get("cta") or "")

        joined = "\n".join([title, hook, body, cta]).strip()
        if not hook or not body:
            continue
        if any(_similarity(joined, existing) >= 0.76 for existing in seen):
            continue

        seen.append(joined)
        score = _score_repurpose_item("\n".join([hook, body, cta]))
        normalized.append(
            {
                "content_type": content_type,
                "title": title,
                "hook": hook,
                "body": body,
                "cta": cta,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_trend_to_content_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        format_name = _clean_text(item.get("format_name") or f"Trend Output {index + 1}")
        hook = _clean_text(item.get("hook") or "")
        body = _postprocess_general_text(item.get("body") or "")
        cta = _clean_text(item.get("cta") or "")

        joined = "\n".join([format_name, hook, body, cta]).strip()
        if not hook or not body:
            continue
        if any(_similarity(joined, existing) >= 0.76 for existing in seen):
            continue

        seen.append(joined)
        score = _score_repurpose_item("\n".join([hook, body, cta]))
        normalized.append(
            {
                "format_name": format_name,
                "hook": hook,
                "body": body,
                "cta": cta,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_nigerian_rewrite_items(items_raw: Any, *, original_text: str) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if isinstance(item, dict):
            version_name = _clean_text(item.get("version_name") or f"Alternative {index + 1}")
            text = _postprocess_humanizer_text(item.get("text") or "")
            why = _clean_text(item.get("why") or "")
        else:
            version_name = f"Alternative {index + 1}"
            text = _postprocess_humanizer_text(str(item))
            why = ""

        if not text:
            continue
        if any(_similarity(text, existing) >= 0.84 for existing in seen):
            continue

        seen.append(text)
        score = _score_humanizer_text(text, original_text, True)
        normalized.append(
            {
                "version_name": version_name,
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


def _normalize_brand_voice_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        voice_name = _clean_text(item.get("voice_name") or f"Voice {index + 1}")
        voice_summary = _postprocess_general_text(item.get("voice_summary") or "")
        example_output = _postprocess_general_text(item.get("example_output") or "")
        guidance = _postprocess_general_text(item.get("guidance") or "")

        joined = "\n".join([voice_name, voice_summary, example_output, guidance]).strip()
        if not voice_summary or not example_output:
            continue
        if any(_similarity(joined, existing) >= 0.78 for existing in seen):
            continue

        seen.append(joined)
        score = _score_repurpose_item("\n".join([voice_summary, example_output, guidance]))
        normalized.append(
            {
                "voice_name": voice_name,
                "voice_summary": voice_summary,
                "example_output": example_output,
                "guidance": guidance,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _normalize_content_differentiation_items(items_raw: Any) -> list[dict[str, Any]]:
    normalized = []
    seen: list[str] = []

    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue

        variant_name = _clean_text(item.get("variant_name") or f"Differentiation {index + 1}")
        differentiator = _postprocess_general_text(item.get("differentiator") or "")
        opportunity = _postprocess_general_text(item.get("opportunity") or "")
        example = _postprocess_general_text(item.get("example") or "")

        joined = "\n".join([variant_name, differentiator, opportunity, example]).strip()
        if not differentiator or not opportunity:
            continue
        if any(_similarity(joined, existing) >= 0.78 for existing in seen):
            continue

        seen.append(joined)
        score = _score_repurpose_item("\n".join([differentiator, opportunity, example]))
        normalized.append(
            {
                "variant_name": variant_name,
                "differentiator": differentiator,
                "opportunity": opportunity,
                "example": example,
                "score": score,
                "label": _title_label(score),
            }
        )

        if len(normalized) >= 3:
            break

    normalized.sort(key=lambda item: item["score"], reverse=True)
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
        logger.warning(
            "Primary %s generation failed. Retrying with secondary OpenAI prompt. Reason: %s",
            tool_name,
            str(primary_exc),
        )

        try:
            raw = _call_json_model(
                instructions=secondary_instructions,
                prompt=secondary_prompt,
                max_output_tokens=max_output_tokens,
            )
            normalized = normalize_fn(raw)
            count = len(normalized) if isinstance(normalized, list) else 1
            logger.info(
                "Secondary %s path produced %s distinct items before final check.",
                tool_name,
                count,
            )

            if count >= min_secondary:
                logger.info("%s used secondary OpenAI path successfully.", tool_name)
                return normalized

            raise ValueError(
                f"Secondary {tool_name} generation returned too few distinct items: {count}"
            )

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
    template_guidance = HOOK_TEMPLATE_GUIDANCE.get(
        template_key,
        HOOK_TEMPLATE_GUIDANCE["curiosity"],
    )
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


def generate_hook_angles(
    topic: str,
    *,
    platform: str = "General",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    banned_phrases = _parse_csvish(avoid_phrases)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "brand_voice": brand_voice,
        "avoid_phrases": banned_phrases,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You generate strategic content angles in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 5 angles under "angles".
- Each angle must be meaningfully different.
- Cover these angle types when possible: {", ".join(HOOK_ANGLE_TYPES)}.
- Keep the angles specific, believable, and useful for creators.
- Do not invent fake proof, fake stories, or exaggerated claims.
- Avoid generic filler and vague advice.
- Avoid these banned phrases when possible: {", ".join(banned_phrases) if banned_phrases else "None"}.
- Each angle item must contain:
  - angle_type
  - angle_title
  - explanation
  - sample_hook
  - best_use_case
- sample_hook should be short, human, and hook-ready.
- angle_title should be concise and strong.
"""

    primary_prompt = f"""
Create 5 content angles in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger hook angles in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 5 angles under "angles".
- Make every angle clearly distinct in framing.
- Prioritize strong positioning, better scroll-stopping contrast, and creator usefulness.
- Avoid fake guru language, generic templates, and unsupported hype.
- Avoid these banned phrases when possible: {", ".join(banned_phrases) if banned_phrases else "None"}.
- Each angle item must contain:
  - angle_type
  - angle_title
  - explanation
  - sample_hook
  - best_use_case
"""

    secondary_prompt = f"""
Generate 5 stronger content angles in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Brand voice: {brand_voice or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        angles = _normalize_hook_angle_items(
            raw.get("angles", []) if isinstance(raw, dict) else []
        )

        if banned_phrases:
            filtered = []
            for angle_item in angles:
                joined = " ".join(
                    [
                        angle_item.get("angle_title", ""),
                        angle_item.get("explanation", ""),
                        angle_item.get("sample_hook", ""),
                        angle_item.get("best_use_case", ""),
                    ]
                ).lower()

                if any(bad.lower() in joined for bad in banned_phrases if bad.strip()):
                    continue
                filtered.append(angle_item)

            if filtered:
                return filtered

        return angles

    return _generate_with_retry(
        tool_name="generate_hook_angles",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_hook_angles(topic, normalized_platform),
        max_output_tokens=1800,
    )


def generate_ctas(
    topic: str,
    platform: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    tone: str | None = None,
    brand_voice: str | None = None,
    avoid_phrases: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)
    support_first_person = _supports_first_person_context(
        offer=offer,
        extra_context=extra_context,
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
        "tone": tone,
        "brand_voice": brand_voice,
        "avoid_phrases": _parse_csvish(avoid_phrases),
        "extra_context": extra_context,
    }

    first_person_rule = (
        "First-person CTAs are allowed only if the user clearly provided personal-story context."
        if support_first_person
        else "Do not use first-person story or confession framing."
    )

    primary_instructions = f"""
You write strong call-to-actions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 CTA options under "ctas".
- Each CTA must be meaningfully different.
- Use these styles when possible: {", ".join(CTA_STYLES)}.
- Keep them natural, actionable, and platform-aware.
- Match the audience, goal, tone, and offer.
- Do not sound robotic, spammy, or overly salesy.
- Avoid fake urgency, vague hype, and generic filler.
- {first_person_rule}
- Each item must contain:
  - text
  - style
  - why
"""

    primary_prompt = f"""
Create 4 CTA options in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You write stronger CTAs in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 CTAs under "ctas".
- Make each CTA distinct in intent and phrasing.
- Prioritize clarity, action, and believable value.
- Avoid generic marketing language, fake urgency, and empty motivation.
- {first_person_rule}
- Each item must contain:
  - text
  - style
  - why
"""

    secondary_prompt = f"""
Generate 4 stronger CTA options in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Tone: {tone or ""}
Brand voice: {brand_voice or ""}
Avoid phrases: {avoid_phrases or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_cta_items(
            raw.get("ctas", []) if isinstance(raw, dict) else [],
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="generate_ctas",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_ctas(topic, normalized_platform),
        max_output_tokens=1400,
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


def generate_repurpose_outputs(
    source_text: str,
    *,
    topic: str | None = None,
    platform: str = "General",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    source_clean = _clean_multiline(source_text)
    if not source_clean:
        raise HTTPException(status_code=400, detail="Source text is required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "source_text": source_clean,
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You repurpose one source idea into multiple content outputs in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 outputs under "outputs".
- Each item must contain:
  - format_name
  - text
  - why
- Make each output clearly different in structure and use case.
- Keep the core meaning of the source text intact.
- Do not invent fake proof, testimonials, or exaggerated claims.
- Make the rewrites more useful, cleaner, and more engaging.
"""

    primary_prompt = f"""
Create 4 repurposed outputs in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger repurposed content outputs in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 outputs under "outputs".
- Make the outputs distinct in format and angle.
- Keep them human, useful, and platform-aware.
- Avoid generic filler, fake urgency, and hype-heavy marketing language.
- Each item must contain:
  - format_name
  - text
  - why
"""

    secondary_prompt = f"""
Generate 4 stronger repurposed outputs in json.

Source text:
{source_clean}

Topic: {topic or ""}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_repurpose_items(raw.get("outputs", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_repurpose_outputs",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_repurpose_outputs(topic or "", normalized_platform, source_clean),
        max_output_tokens=1800,
    )


def generate_ad_copy_variants(
    topic: str,
    *,
    platform: str = "Facebook",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You generate ad copy variants in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 variants under "variants".
- Each item must contain:
  - variant_name
  - primary_text
  - headline
  - description
  - cta
- Keep the ads clear, believable, and conversion-aware.
- Avoid fake claims, unrealistic outcomes, fake urgency, and spammy copy.
- Make each variant meaningfully different in angle.
"""

    primary_prompt = f"""
Create 3 ad copy variants in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger ad copy variants in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 variants under "variants".
- Make each variant distinct in angle and message.
- Keep the copy natural, focused, and ad-ready.
- Avoid generic marketing filler, fake proof, and exaggerated results.
- Each item must contain:
  - variant_name
  - primary_text
  - headline
  - description
  - cta
"""

    secondary_prompt = f"""
Generate 3 stronger ad copy variants in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_ad_copy_items(raw.get("variants", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_ad_copy_variants",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_ad_copy(topic, normalized_platform, offer),
        max_output_tokens=1600,
    )


def generate_carousel_outlines(
    topic: str,
    *,
    platform: str = "Instagram",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    angle: str | None = None,
    style: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "angle": angle,
        "content_type": style,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You create carousel outlines in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 carousel outlines under "carousels".
- Each item must contain:
  - title
  - style
  - hook_slide
  - slides
  - closing_slide
  - caption
- slides must be an array.
- Each carousel should feel distinct in structure and angle.
- Keep the wording concise and useful for slide-based content.
- Avoid generic fluff, fake proof, and exaggerated claims.
"""

    primary_prompt = f"""
Create 3 carousel outlines in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger carousel outlines in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 carousel outlines under "carousels".
- Make each one structurally distinct and easy to turn into slides.
- Use sharper hooks, clearer steps, and more useful takeaways.
- Avoid generic marketing language and hype-heavy phrasing.
- Each item must contain:
  - title
  - style
  - hook_slide
  - slides
  - closing_slide
  - caption
"""

    secondary_prompt = f"""
Generate 3 stronger carousel outlines in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Angle: {angle or ""}
Style: {style or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_carousel_items(raw.get("carousels", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_carousel_outlines",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_carousels(topic, normalized_platform, style),
        max_output_tokens=1800,
    )


def generate_offer_positioning(
    topic: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    context = {
        "topic": topic,
        "audience": audience,
        "goal": goal,
        "offer": offer,
        "pain_point": pain_point,
        "desired_outcome": desired_outcome,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You generate positioning options in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 positioning variants under "positioning".
- Each item must contain:
  - variant_name
  - positioning_angle
  - value_proposition
  - who_its_for
  - problem_frame
  - differentiator
  - message_example
- Make each variant clearly distinct.
- Keep them specific, believable, and useful for product messaging.
- Avoid vague fluff, generic hype, and unrealistic outcomes.
"""

    primary_prompt = f"""
Create 3 offer positioning variants in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger offer positioning options in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 variants under "positioning".
- Make each variant meaningfully different in angle and message.
- Keep the positioning sharper, clearer, and more differentiated.
- Avoid generic startup language, fake proof, and exaggerated claims.
- Each item must contain:
  - variant_name
  - positioning_angle
  - value_proposition
  - who_its_for
  - problem_frame
  - differentiator
  - message_example
"""

    secondary_prompt = f"""
Generate 3 stronger offer positioning variants in json.

Topic: {topic}
Audience: {audience or ""}
Goal: {goal or ""}
Offer: {offer or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_positioning_items(raw.get("positioning", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_offer_positioning",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_offer_positioning(topic, offer),
        max_output_tokens=1800,
    )


def generate_viral_rewrites(
    source_text: str,
    *,
    topic: str | None = None,
    language: str = "en",
    platform: str = "General",
    audience: str | None = None,
    goal: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    source_clean = _clean_multiline(source_text)
    if not source_clean:
        raise HTTPException(status_code=400, detail="Source text is required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "source_text": source_clean,
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You rewrite content to feel more viral in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrites under "rewrites".
- Each item must contain:
  - version_name
  - text
  - why
- Keep the core meaning intact while making the opening, tension, and phrasing stronger.
- Do not use fake proof, exaggerated claims, or spammy clickbait.
- Make the rewrites feel more human and more scroll-stopping.
"""

    primary_prompt = f"""
Create 4 viral rewrites in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger viral-style rewrites in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrites under "rewrites".
- Make them more engaging, more tension-led, and more platform-aware.
- Avoid fake urgency, fake story claims, and overhyped marketing language.
- Each item must contain:
  - version_name
  - text
  - why
"""

    secondary_prompt = f"""
Generate 4 stronger viral rewrites in json.

Source text:
{source_clean}

Topic: {topic or ""}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_humanizer_items(
            raw.get("rewrites", []) if isinstance(raw, dict) else [],
            original_text=source_clean,
            preserve_original_meaning=True,
        )

    return _generate_with_retry(
        tool_name="generate_viral_rewrites",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_viral_rewrites(source_clean, topic),
        max_output_tokens=2200,
    )


def analyze_hook_score(
    hook_text: str,
    *,
    platform: str = "General",
    language: str = "en",
    topic: str | None = None,
    audience: str | None = None,
    goal: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    hook_clean = _postprocess_hook_text(hook_text)
    if not hook_clean:
        raise HTTPException(status_code=400, detail="Hook text is required.")

    normalized_platform = _normalize_platform(platform)
    support_first_person = _supports_first_person_context(extra_context=extra_context)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "extra_context": extra_context,
        "hook_text": hook_clean,
    }

    primary_instructions = f"""
You analyze hook quality in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return an object with:
  - hook_text
  - overall_score
  - verdict
  - dimensions
  - strengths
  - weaknesses
  - improved_hook
- dimensions must include:
  - clarity
  - curiosity
  - specificity
  - believability
  - platform_fit
- Scores must be 0-100 style scores.
- Keep the analysis practical and clear.
- Do not give fluffy feedback.
- improved_hook should keep the original intent but make it stronger.
"""

    primary_prompt = f"""
Analyze this hook in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You give practical hook feedback in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return:
  - hook_text
  - overall_score
  - verdict
  - dimensions
  - strengths
  - weaknesses
  - improved_hook
- Keep feedback direct, useful, and platform-aware.
- improved_hook should feel more specific and more compelling without becoming fake or exaggerated.
"""

    secondary_prompt = f"""
Analyze and improve this hook in json.

Hook text: {hook_clean}
Platform: {normalized_platform}
Topic: {topic or ""}
Audience: {audience or ""}
Goal: {goal or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_hook_score_result(
            raw,
            hook_clean,
            platform=normalized_platform,
            support_first_person=support_first_person,
        )

    return _generate_with_retry(
        tool_name="analyze_hook_score",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=1,
        min_secondary=1,
        fallback_fn=lambda: _fallback_hook_score_analysis(
            hook_clean,
            platform=normalized_platform,
            support_first_person=support_first_person,
        ),
        max_output_tokens=1400,
    )


def generate_retention_intros(
    topic: str,
    *,
    platform: str = "YouTube",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    angle: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "angle": angle,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You create retention-focused intros in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 intros under "intros".
- Each item must contain:
  - version_name
  - hook_line
  - preview_line
  - lesson_line
  - cta_line
- The intro should feel like the first 10-20 seconds of a strong video.
- It should create tension, preview value, and make the viewer want to keep watching.
- Avoid generic hype and vague motivational fluff.
"""

    primary_prompt = f"""
Create 3 retention intros in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger video intros in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 intros under "intros".
- Each intro should feel sharper, more watchable, and more structured for retention.
- Use cleaner tension and clearer payoff.
- Avoid generic openers and fake urgency.
- Each item must contain:
  - version_name
  - hook_line
  - preview_line
  - lesson_line
  - cta_line
"""

    secondary_prompt = f"""
Generate 3 stronger retention intros in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Angle: {angle or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_retention_intro_items(raw.get("intros", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_retention_intros",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_retention_intros(topic, normalized_platform),
        max_output_tokens=1600,
    )


def generate_comment_to_content(
    comment_text: str,
    *,
    platform: str = "Instagram",
    language: str = "en",
    topic: str | None = None,
    audience: str | None = None,
    goal: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    comment_clean = _clean_multiline(comment_text)
    if not comment_clean:
        raise HTTPException(status_code=400, detail="Comment text is required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "source_text": comment_clean,
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You turn audience comments into content ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 ideas under "ideas".
- Each item must contain:
  - content_type
  - title
  - hook
  - body
  - cta
- Make the outputs useful and distinct.
- Treat the comment like a signal of what the audience wants more clarity on.
- Avoid generic filler.
"""

    primary_prompt = f"""
Create 3 content ideas from this comment in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You generate stronger comment-to-content ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 ideas under "ideas".
- Make each idea feel distinct in format and angle.
- Keep the outputs practical and creator-ready.
- Each item must contain:
  - content_type
  - title
  - hook
  - body
  - cta
"""

    secondary_prompt = f"""
Generate 3 stronger content ideas from this comment in json.

Comment:
{comment_clean}

Topic: {topic or ""}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_comment_to_content_items(raw.get("ideas", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_comment_to_content",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_comment_to_content(comment_clean, normalized_platform),
        max_output_tokens=1600,
    )


def generate_trend_to_content(
    trend_input: str,
    *,
    platform: str = "TikTok",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    trend_clean = _clean_multiline(trend_input)
    if not trend_clean:
        raise HTTPException(status_code=400, detail="Trend input is required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "source_text": trend_clean,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You turn trends into stronger content ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 outputs under "outputs".
- Each item must contain:
  - format_name
  - hook
  - body
  - cta
- Do not just repeat the trend.
- Reframe the trend into a more useful, more original content angle.
- Avoid generic trend-chasing phrasing.
"""

    primary_prompt = f"""
Create 3 trend-to-content outputs in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You generate stronger trend-based content ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 outputs under "outputs".
- Make each one more useful, more original, and more creator-ready.
- Avoid copying the trend too literally.
- Each item must contain:
  - format_name
  - hook
  - body
  - cta
"""

    secondary_prompt = f"""
Generate 3 stronger trend-to-content outputs in json.

Trend input:
{trend_clean}

Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_trend_to_content_items(raw.get("outputs", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_trend_to_content",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_trend_to_content(trend_clean, normalized_platform),
        max_output_tokens=1600,
    )


def generate_nigerian_audience_rewrites(
    source_text: str,
    *,
    language: str = "en",
    platform: str = "General",
    audience: str | None = None,
    goal: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    source_clean = _clean_multiline(source_text)
    if not source_clean:
        raise HTTPException(status_code=400, detail="Source text is required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "source_text": source_clean,
        "platform": normalized_platform,
        "audience": audience or "Nigerian audience",
        "goal": goal,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You rewrite content for a Nigerian audience in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrites under "rewrites".
- Each item must contain:
  - version_name
  - text
  - why
- Make the copy feel more locally natural and relatable for Nigerians.
- Do not force slang into every line.
- Keep the meaning intact.
- Avoid stereotypes and awkward exaggeration.
"""

    primary_prompt = f"""
Create 4 Nigerian-audience rewrites in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger locally-relevant rewrites for a Nigerian audience in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 rewrites under "rewrites".
- Make them feel more natural, more culturally familiar, and still clear.
- Avoid overdoing slang or making the copy unreadable.
- Each item must contain:
  - version_name
  - text
  - why
"""

    secondary_prompt = f"""
Generate 4 stronger Nigerian-audience rewrites in json.

Source text:
{source_clean}

Platform: {normalized_platform}
Audience: {audience or "Nigerian audience"}
Goal: {goal or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_nigerian_rewrite_items(
            raw.get("rewrites", []) if isinstance(raw, dict) else [],
            original_text=source_clean,
        )

    return _generate_with_retry(
        tool_name="generate_nigerian_audience_rewrites",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=3,
        min_secondary=2,
        fallback_fn=lambda: _fallback_nigerian_rewrites(source_clean),
        max_output_tokens=2000,
    )


def generate_brand_voice_training(
    topic: str,
    brand_samples: str,
    *,
    language: str = "en",
    platform: str = "General",
    audience: str | None = None,
    goal: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    topic_clean = _clean_text(topic)
    samples_clean = _clean_multiline(brand_samples)
    if not topic_clean:
        raise HTTPException(status_code=400, detail="Topic is required.")
    if not samples_clean:
        raise HTTPException(status_code=400, detail="Brand samples are required.")

    normalized_platform = _normalize_platform(platform)

    context = {
        "topic": topic_clean,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "extra_context": extra_context,
        "source_text": samples_clean,
    }

    primary_instructions = f"""
You analyze and train a brand voice in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 voice outputs under "voices".
- Each item must contain:
  - voice_name
  - voice_summary
  - example_output
  - guidance
- The outputs should feel grounded in the writing samples.
- Make the voice advice practical and usable.
- Avoid generic "authentic, engaging, powerful" fluff.
"""

    primary_prompt = f"""
Create 3 brand voice outputs in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You create stronger brand voice guidance in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 outputs under "voices".
- Make each voice variation distinct and practical.
- Keep the guidance specific to the sample style and the topic.
- Each item must contain:
  - voice_name
  - voice_summary
  - example_output
  - guidance
"""

    secondary_prompt = f"""
Generate 3 stronger brand voice outputs in json.

Topic: {topic_clean}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Brand samples:
{samples_clean}

Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_brand_voice_items(raw.get("voices", []) if isinstance(raw, dict) else [])

    return _generate_with_retry(
        tool_name="generate_brand_voice_training",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_brand_voice_outputs(topic_clean, samples_clean),
        max_output_tokens=1800,
    )


def generate_content_differentiation(
    topic: str,
    *,
    language: str = "en",
    platform: str = "General",
    audience: str | None = None,
    goal: str | None = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
) -> list[dict[str, Any]]:
    normalized_platform = _normalize_platform(platform)

    context = {
        "topic": topic,
        "platform": normalized_platform,
        "audience": audience,
        "goal": goal,
        "brand_voice": brand_voice,
        "extra_context": extra_context,
    }

    primary_instructions = f"""
You generate content differentiation ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 items under "differentiation".
- Each item must contain:
  - variant_name
  - differentiator
  - opportunity
  - example
- Focus on how to stand out without becoming fake or louder for no reason.
- Keep the advice specific and practical.
"""

    primary_prompt = f"""
Create 3 content differentiation ideas in json.

Context:
{_context_summary(context)}
"""

    secondary_instructions = f"""
You generate stronger differentiation ideas in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 3 items under "differentiation".
- Make each one distinct and practical.
- Avoid vague brand-strategy fluff.
- Each item must contain:
  - variant_name
  - differentiator
  - opportunity
  - example
"""

    secondary_prompt = f"""
Generate 3 stronger content differentiation ideas in json.

Topic: {topic}
Platform: {normalized_platform}
Audience: {audience or ""}
Goal: {goal or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}
"""

    def normalize_fn(raw):
        return _normalize_content_differentiation_items(
            raw.get("differentiation", []) if isinstance(raw, dict) else []
        )

    return _generate_with_retry(
        tool_name="generate_content_differentiation",
        primary_instructions=primary_instructions,
        primary_prompt=primary_prompt,
        secondary_instructions=secondary_instructions,
        secondary_prompt=secondary_prompt,
        normalize_fn=normalize_fn,
        min_primary=2,
        min_secondary=2,
        fallback_fn=lambda: _fallback_content_differentiation(topic, audience),
        max_output_tokens=1600,
    )