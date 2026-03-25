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
    "curiosity",
    "contrarian",
    "mistake",
    "result",
    "confession",
    "warning",
]

TITLE_ARCHETYPES = [
    "search-led",
    "contrarian",
    "mistake-led",
    "result-led",
    "curiosity-led",
    "proof-led",
]

CAPTION_STYLES = [
    "story",
    "authority",
    "community",
    "urgency",
]

SCRIPT_STYLES = [
    "direct-to-camera",
    "story-led",
    "problem-solution",
    "myth-busting",
]


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
    # 1. Try the convenience field first.
    output_text = _obj_get(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # 2. Walk the output/content structure robustly.
    output = _obj_get(response, "output", []) or []
    collected: list[str] = []

    for item in output:
        # Some SDK responses return message items with nested content.
        content = _obj_get(item, "content", []) or []

        for part in content:
            part_type = _obj_get(part, "type", None)

            # Common documented case: output_text with a text field.
            if part_type == "output_text":
                text_value = _obj_get(part, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value.strip())
                    continue

            # Broader fallback handling in case SDK shape differs.
            text_value = _obj_get(part, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                collected.append(text_value.strip())
                continue

            # Sometimes text may be nested one level deeper.
            nested_text = _obj_get(text_value, "value", None)
            if isinstance(nested_text, str) and nested_text.strip():
                collected.append(nested_text.strip())
                continue

        # Some response items may expose text directly.
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

    return "\n".join(lines).strip()


def _responses_create(*, instructions: str, input_text: str, max_output_tokens: int = 1200) -> Any:
    _ensure_openai_ready()
    return client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_object"
            }
        },
    )


def _call_json_model(
    *,
    instructions: str,
    prompt: str,
    max_output_tokens: int = 1200,
) -> Any:
    """
    1. Ask the model for JSON text output explicitly.
    2. Try parsing directly.
    3. If parsing fails, ask the model once to repair its own output into valid JSON.
    """
    _ensure_openai_ready()

    raw_text = ""
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


def _score_hook_text(text: str) -> int:
    cleaned = _clean_text(text)
    words = [w for w in cleaned.split(" ") if w]
    word_count = len(words)
    char_count = len(cleaned)

    curiosity_words = {
        "why", "how", "secret", "mistake", "truth", "stop", "best", "worst",
        "nobody", "everyone", "before", "after", "never", "real"
    }
    emotional_words = {
        "dangerous", "powerful", "simple", "shocking", "hard", "easy",
        "hidden", "real", "smart", "wrong", "brutal", "fast"
    }

    brevity = 28 if 6 <= word_count <= 14 else 20 if 4 <= word_count <= 16 else 12
    readability = 24 if char_count <= 90 else 18 if char_count <= 120 else 12
    curiosity = 24 if any(w.lower().strip("?!.,:;") in curiosity_words for w in words) else 14
    punch = 24 if any(w.lower().strip("?!.,:;") in emotional_words for w in words) else 14

    return _clamp(brevity + readability + curiosity + punch, 35, 98)


def _score_title_text(text: str) -> int:
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

    if re.search(r"\b(how|why|what|best|mistake|secret|truth)\b", cleaned, re.IGNORECASE):
        score += 10
    if re.search(r"[:?!]", cleaned):
        score += 6

    return _clamp(score, 40, 98)


def _score_caption_text(text: str) -> int:
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

    return _clamp(score, 40, 98)


def _score_description_text(text: str) -> int:
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

    return _clamp(score, 40, 98)


def _score_script_text(text: str) -> int:
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


def _score_thumbnail_text(text: str) -> int:
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

    return _clamp(score, 35, 98)


def _title_label(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Decent"
    return "Weak"


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
            "score": _score_hook_text(text),
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
        f"If your {topic} feels repetitive, the problem usually is not effort. It is weak positioning.\n\nA stronger angle makes people stop, read, and care.\n\nComment \"part 2\" if you want more.",
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
        normalized.append(
            {
                **item,
                "score": score,
                "label": _title_label(score),
            }
        )
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


def _normalize_scored_text_items(
    value: Any,
    *,
    limit: int,
    score_fn,
) -> list[dict[str, Any]]:
    items: list[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("title") or item.get("hook") or ""
                items.append(str(text))
            else:
                items.append(str(item))
    else:
        raise ValueError("Expected list response.")

    unique = _dedupe_text_items(items, limit=limit, min_items=1)

    normalized: list[dict[str, Any]] = []
    for text in unique:
        score = score_fn(text)
        normalized.append(
            {
                "text": text,
                "score": score,
                "label": _title_label(score),
            }
        )
    return normalized


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

    instructions = f"""
You create high-performing YouTube titles in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 6 title options in a JSON object under "titles".
- Make each title meaningfully different in structure, not just word swaps.
- Spread across these archetypes: {", ".join(TITLE_ARCHETYPES)}.
- Keep titles natural, specific, and varied.
- Avoid repetitive openers and repetitive syntax.
- Use keywords naturally, not awkwardly.
- Make them feel tailored to the user's actual audience and goal.
- Do not add numbering in title text.
- Each item should be an object with:
  - text
  - archetype
  - keyword_focus
"""
    prompt = f"""
Create 6 distinct YouTube titles.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=900,
        )

        titles = raw.get("titles", [])
        normalized = []
        seen: list[str] = []

        for item in titles:
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
            score = _score_title_text(text)
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

        if len(normalized) < 4:
            raise ValueError("Too few distinct titles returned.")

        return normalized
    except Exception as exc:
        logger.warning("generate_titles fell back. Reason: %s", str(exc))
        return _fallback_titles(topic)


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

    instructions = f"""
You write platform-native social captions in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 4 caption options under "captions".
- Each caption must feel meaningfully different in voice and structure.
- Use these style buckets once each when possible: {", ".join(CAPTION_STYLES)}.
- Adapt to the user's platform, goal, and audience.
- Use a CTA naturally. Avoid robotic wording.
- Do not reuse the same first sentence pattern.
- Each item should be an object with:
  - text
  - style
  - cta
"""
    prompt = f"""
Create 4 distinct social captions.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=1300,
        )

        captions = raw.get("captions", [])
        normalized = []
        seen: list[str] = []

        for item in captions:
            if isinstance(item, dict):
                text = _clean_multiline(item.get("text") or "")
                style = _clean_text(item.get("style") or "general")
                cta = _clean_text(item.get("cta") or "")
            else:
                text = _clean_multiline(str(item))
                style = "general"
                cta = ""

            if not text:
                continue
            if any(_similarity(text, existing) >= 0.72 for existing in seen):
                continue

            seen.append(text)
            score = _score_caption_text(text)
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

        if len(normalized) < 3:
            raise ValueError("Too few distinct captions returned.")

        return normalized
    except Exception as exc:
        logger.warning("generate_captions fell back. Reason: %s", str(exc))
        return _fallback_captions(topic, normalized_platform, tone)


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

    instructions = f"""
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
    prompt = f"""
Create 3 distinct hashtag sets.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=900,
        )

        sets = raw.get("sets", [])
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(sets):
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

        if len(normalized) < 2:
            raise ValueError("Too few valid hashtag sets returned.")

        return normalized
    except Exception as exc:
        logger.warning("generate_hashtags fell back. Reason: %s", str(exc))
        return _fallback_hashtags(topic, normalized_platform)


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

    instructions = f"""
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
"""
    prompt = f"""
Create 3 description rewrites.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=1400,
        )

        rewrites = raw.get("rewrites", [])
        normalized = []
        seen: list[str] = []

        for item in rewrites:
            if isinstance(item, dict):
                title = _clean_text(item.get("title") or "Rewrite")
                text = _clean_multiline(item.get("text") or "")
            else:
                title = "Rewrite"
                text = _clean_multiline(str(item))

            if not text:
                continue
            if any(_similarity(text, existing) >= 0.75 for existing in seen):
                continue

            seen.append(text)
            score = _score_description_text(text)
            normalized.append(
                {
                    "title": title,
                    "text": text,
                    "score": score,
                    "label": _title_label(score),
                    "platform": normalized_platform,
                }
            )

            if len(normalized) >= 3:
                break

        if len(normalized) < 2:
            raise ValueError("Too few distinct rewrites returned.")

        return normalized
    except Exception as exc:
        logger.warning("generate_description_rewrites fell back. Reason: %s", str(exc))
        return _fallback_descriptions(topic, normalized_platform)


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

    instructions = f"""
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
- body should be concise but complete.
- Do not make all scripts sound like the same template with one word changed.
"""
    prompt = f"""
Create 3 distinct short-form scripts.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=1500,
        )

        scripts = raw.get("scripts", [])
        normalized = []
        seen: list[str] = []

        for index, item in enumerate(scripts):
            if not isinstance(item, dict):
                continue

            hook = _clean_text(item.get("hook") or "")
            body = _clean_multiline(item.get("body") or "")
            cta = _clean_text(item.get("cta") or "")
            title = _clean_text(item.get("title") or f"Script Draft {index + 1}")
            style_name = _clean_text(item.get("style") or "general")

            text = f"Hook: {hook}\n\nBody: {body}\n\nCTA: {cta}".strip()
            if not hook or not body:
                continue
            if any(_similarity(text, existing) >= 0.74 for existing in seen):
                continue

            seen.append(text)
            score = _score_script_text(text)
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
                    "platform": normalized_platform,
                }
            )

            if len(normalized) >= 3:
                break

        if len(normalized) < 2:
            raise ValueError("Too few distinct scripts returned.")

        return normalized
    except Exception as exc:
        logger.warning("generate_scripts fell back. Reason: %s", str(exc))
        return _fallback_scripts(topic, normalized_platform)


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

    instructions = f"""
You analyze and improve thumbnail text in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return:
  - analysis: object with text, feedback, metrics
  - suggestions: array of 5 improved thumbnail text options
- Thumbnail text should stay short, clear, and clickable.
- Avoid full-sentence clutter.
- Suggestions must be distinct, not minor rewrites of each other.
"""
    prompt = f"""
Analyze thumbnail text and suggest better options.

Context:
{_context_summary(context)}
"""
    try:
        raw = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=950,
        )

        analysis_raw = raw.get("analysis", {}) if isinstance(raw, dict) else {}
        text_value = _clean_text(
            analysis_raw.get("text")
            or thumbnail_text
            or topic
        )
        if not text_value:
            raise ValueError("Missing thumbnail text.")

        score = _score_thumbnail_text(text_value)
        feedback = analysis_raw.get("feedback") or []
        if not isinstance(feedback, list):
            feedback = []

        suggestions_raw = raw.get("suggestions", []) if isinstance(raw, dict) else []
        suggestions = _normalize_scored_text_items(
            suggestions_raw,
            limit=5,
            score_fn=_score_thumbnail_text,
        )

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
    except Exception as exc:
        logger.warning("analyze_thumbnail_text fell back. Reason: %s", str(exc))
        return _fallback_thumbnail(topic, thumbnail_text)


def _normalize_hook_items(
    hooks_raw: Any,
    *,
    normalized_platform: str,
    banned_texts: list[str],
    template_key: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    accepted_texts: list[str] = []

    for item in hooks_raw:
        if isinstance(item, dict):
            text = _clean_text(item.get("text") or "")
            family = _clean_text(item.get("family") or template_key)
        else:
            text = _clean_text(str(item))
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
                "score": _score_hook_text(text),
                "platform": normalized_platform,
                "family": family,
            }
        )

        if len(normalized) >= 6:
            break

    return normalized


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
    template_key = _clean_text(template or "curiosity").lower()
    template_guidance = HOOK_TEMPLATE_GUIDANCE.get(template_key, HOOK_TEMPLATE_GUIDANCE["curiosity"])
    banned_texts = [_clean_text(item) for item in (avoid_hooks or []) if _clean_text(item)]

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

    primary_instructions = f"""
You generate viral-style content hooks in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 6 hooks under "hooks".
- Each hook must be distinctly different in structure.
- Cover these families when possible: {", ".join(HOOK_FAMILIES)}.
- Template guidance: {template_guidance}
- Avoid generic sameness.
- Avoid sounding AI-generated or overstuffed.
- Keep hooks concise and punchy.
- Do not reuse the same opener pattern.
- Do not simply prepend the topic to stock templates.
- Write hooks that feel native, specific, and human.
- Each item should be an object with:
  - text
  - family
"""
    primary_prompt = f"""
Create 6 hook options.

Context:
{_context_summary(context)}

Avoid hooks or phrases too similar to these:
{json.dumps(banned_texts[:20], ensure_ascii=False)}
"""

    secondary_instructions = f"""
You write high-performing short hooks in {_language_name(language)}.
Return valid JSON only.

Rules:
- Return exactly 6 hooks in an object under "hooks".
- Hooks must feel specific to the topic, not like templates with the topic pasted in.
- Every hook must use a different structure and opening.
- Avoid phrases like:
  - "Nobody talks about this part of..."
  - "What actually works for..."
  - "Before you post about..."
unless they are truly the best fit.
- Make the writing feel original and human.
- Each hook item must contain:
  - text
  - family
"""
    secondary_prompt = f"""
Generate 6 stronger hook options for this topic.

Topic: {topic}
Platform: {normalized_platform}
Template: {template_key}
Audience: {audience or ""}
Goal: {goal or ""}
Pain point: {pain_point or ""}
Desired outcome: {desired_outcome or ""}
Angle: {angle or ""}
Keywords: {keywords or ""}
Brand voice: {brand_voice or ""}
Extra context: {extra_context or ""}

Hooks to avoid:
{json.dumps(banned_texts[:20], ensure_ascii=False)}
"""

    try:
        raw = _call_json_model(
            instructions=primary_instructions,
            prompt=primary_prompt,
            max_output_tokens=900,
        )
        hooks_raw = raw.get("hooks", [])
        normalized = _normalize_hook_items(
            hooks_raw,
            normalized_platform=normalized_platform,
            banned_texts=banned_texts,
            template_key=template_key,
        )

        if len(normalized) >= 4:
            logger.info("generate_hooks used primary OpenAI path successfully.")
            return normalized

        raise ValueError("Primary hook generation returned too few distinct hooks.")

    except Exception as primary_exc:
        logger.warning("Primary hook generation failed. Retrying with secondary OpenAI prompt. Reason: %s", str(primary_exc))

        try:
            raw = _call_json_model(
                instructions=secondary_instructions,
                prompt=secondary_prompt,
                max_output_tokens=900,
            )
            hooks_raw = raw.get("hooks", [])
            normalized = _normalize_hook_items(
                hooks_raw,
                normalized_platform=normalized_platform,
                banned_texts=banned_texts,
                template_key=template_key,
            )

            if len(normalized) >= 4:
                logger.info("generate_hooks used secondary OpenAI path successfully.")
                return normalized

            raise ValueError("Secondary hook generation returned too few distinct hooks.")

        except Exception as secondary_exc:
            logger.warning(
                "generate_hooks fell back to emergency templates. Primary reason: %s | Secondary reason: %s",
                str(primary_exc),
                str(secondary_exc),
            )

            fallback = _fallback_hooks(topic, normalized_platform, template_key)
            filtered: list[dict[str, Any]] = []
            for item in fallback:
                text = item["text"]
                if any(_similarity(text, banned) >= 0.72 for banned in banned_texts):
                    continue
                if any(_similarity(text, existing["text"]) >= 0.72 for existing in filtered):
                    continue
                filtered.append(item)

            return filtered[:6] if filtered else fallback[:6]


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