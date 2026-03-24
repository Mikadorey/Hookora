import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from openai import OpenAI

from app.config import settings

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
    "curiosity": "Lead with open loops, tension, surprise, or unfinished information.",
    "authority": "Lead with conviction, insight, proof, or expert positioning.",
    "problem": "Lead with a painful mistake, friction point, or costly misunderstanding.",
    "benefit": "Lead with transformation, payoff, or outcome-first value.",
    "controversy": "Lead with disagreement, contrast, or a strong contrarian angle.",
    "story": "Lead with a moment, situation, confession, or human tension.",
}

DEFAULT_BANNED_OPENERS = {
    "here are",
    "in this video",
    "today we will",
    "let's talk about",
    "welcome back",
    "this post is about",
    "did you know that",
    "if you want to",
    "discover how to",
}

TITLE_ARCHETYPES = [
    "search intent",
    "curiosity gap",
    "mistake-focused",
    "contrarian",
    "outcome-driven",
    "framework/list",
]

HOOK_ARCHETYPES = [
    "question",
    "contrarian",
    "mistake",
    "proof/result",
    "story",
    "warning",
]

SCRIPT_ARCHETYPES = [
    "direct-to-camera",
    "story-led",
    "contrarian explainer",
    "problem-solution",
]

CAPTION_ARCHETYPES = [
    "conversation starter",
    "insight-driven",
    "story-led",
    "cta-heavy",
]

DESCRIPTION_ARCHETYPES = [
    "seo structured",
    "benefit-first",
    "authority-led",
]

HASHTAG_SET_TYPES = [
    "discovery",
    "balanced",
    "niche",
]


def _ensure_openai_ready() -> None:
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is missing. Add OPENAI_API_KEY to your backend environment."
        )


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_language(language: str | None) -> str:
    cleaned = _clean_text(language or "en").lower()
    return cleaned if cleaned in SUPPORTED_LANGUAGES else "en"


def _language_name(language: str | None) -> str:
    return SUPPORTED_LANGUAGES[_normalize_language(language)]


def _normalize_platform(platform: str | None) -> str:
    cleaned = _clean_text(platform or "")
    if not cleaned:
        return "General"
    return SUPPORTED_HOOK_PLATFORMS.get(cleaned.lower(), cleaned)


def _clamp(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _safe_json_load(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty JSON response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return json.loads(fenced_match.group(1).strip())

        raw_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if raw_match:
            return json.loads(raw_match.group(1).strip())

        raise


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    collected: list[str] = []

    for item in output:
        contents = getattr(item, "content", None) or []
        for content in contents:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                collected.append(text_value.strip())

    if collected:
        return "\n".join(collected).strip()

    raise ValueError("Model returned no text output.")


def _call_json_model(
    *,
    instructions: str,
    prompt: str,
    max_output_tokens: int = 1800,
) -> Any:
    _ensure_openai_ready()

    try:
        response = client.responses.create(
            model=settings.openai_model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_object"}},
        )
        text = _extract_output_text(response)
        return _safe_json_load(text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI request failed: {str(exc)}"
        ) from exc


def _normalize_keywords(raw_keywords: Any) -> list[str]:
    if isinstance(raw_keywords, list):
        parts = raw_keywords
    elif isinstance(raw_keywords, str):
        parts = re.split(r"[,|\n]", raw_keywords)
    else:
        parts = []

    cleaned: list[str] = []
    seen: set[str] = set()

    for part in parts:
        text = _clean_text(str(part))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    return cleaned[:10]


def _normalize_avoid_list(raw_items: Any) -> list[str]:
    if isinstance(raw_items, list):
        parts = raw_items
    elif isinstance(raw_items, str):
        parts = re.split(r"[\n|]+", raw_items)
    else:
        parts = []

    cleaned: list[str] = []
    seen: set[str] = set()

    for part in parts:
        text = _clean_text(str(part))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    return cleaned[:12]


def _context_value(context: Dict[str, Any], key: str) -> str:
    value = context.get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_clean_text(str(v)) for v in value if _clean_text(str(v)))
    return _clean_text(str(value))


def _build_context_dict(
    *,
    topic: str,
    platform: str | None = None,
    language: str | None = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
    template: str | None = None,
    tone: str | None = None,
    original_description: str | None = None,
    thumbnail_text: str | None = None,
) -> Dict[str, Any]:
    return {
        "topic": _clean_text(topic),
        "platform": _normalize_platform(platform),
        "language": _normalize_language(language),
        "audience": _clean_text(audience),
        "goal": _clean_text(goal),
        "product_or_offer": _clean_text(product_or_offer),
        "pain_point": _clean_text(pain_point),
        "desired_outcome": _clean_text(desired_outcome),
        "angle": _clean_text(angle),
        "call_to_action": _clean_text(call_to_action),
        "keywords": _normalize_keywords(keywords),
        "brand_voice": _clean_text(brand_voice),
        "content_type": _clean_text(content_type),
        "extra_context": _clean_text(extra_context),
        "examples_to_avoid": _normalize_avoid_list(examples_to_avoid),
        "banned_phrases": _normalize_avoid_list(banned_phrases),
        "template": _clean_text(template),
        "tone": _clean_text(tone),
        "original_description": _clean_text(original_description),
        "thumbnail_text": _clean_text(thumbnail_text),
    }


def _context_lines(context: Dict[str, Any]) -> str:
    lines = [
        f"Language: {_language_name(context.get('language'))}",
        f"Topic: {_context_value(context, 'topic') or 'N/A'}",
        f"Platform: {_context_value(context, 'platform') or 'N/A'}",
        f"Audience: {_context_value(context, 'audience') or 'N/A'}",
        f"Goal: {_context_value(context, 'goal') or 'N/A'}",
        f"Product or offer: {_context_value(context, 'product_or_offer') or 'N/A'}",
        f"Pain point: {_context_value(context, 'pain_point') or 'N/A'}",
        f"Desired outcome: {_context_value(context, 'desired_outcome') or 'N/A'}",
        f"Angle: {_context_value(context, 'angle') or 'N/A'}",
        f"CTA: {_context_value(context, 'call_to_action') or 'N/A'}",
        f"Keywords: {_context_value(context, 'keywords') or 'N/A'}",
        f"Brand voice: {_context_value(context, 'brand_voice') or 'N/A'}",
        f"Content type: {_context_value(context, 'content_type') or 'N/A'}",
        f"Tone: {_context_value(context, 'tone') or 'N/A'}",
        f"Template: {_context_value(context, 'template') or 'N/A'}",
        f"Original description: {_context_value(context, 'original_description') or 'N/A'}",
        f"Thumbnail text: {_context_value(context, 'thumbnail_text') or 'N/A'}",
        f"Extra context: {_context_value(context, 'extra_context') or 'N/A'}",
        f"Examples to avoid: {_context_value(context, 'examples_to_avoid') or 'N/A'}",
        f"Banned phrases: {_context_value(context, 'banned_phrases') or 'N/A'}",
    ]
    return "\n".join(lines)


def _tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (value or "").lower()))


def _text_similarity(a: str, b: str) -> float:
    a_clean = _clean_text(a).lower()
    b_clean = _clean_text(b).lower()
    if not a_clean or not b_clean:
        return 0.0

    seq_ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
    a_tokens = _tokenize(a_clean)
    b_tokens = _tokenize(b_clean)

    if not a_tokens or not b_tokens:
        return seq_ratio

    overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    return max(seq_ratio, overlap)


def _starts_with_banned_opener(text: str, banned_phrases: list[str]) -> bool:
    lower_text = _clean_text(text).lower()
    for phrase in list(DEFAULT_BANNED_OPENERS) + [p.lower() for p in banned_phrases]:
        if phrase and lower_text.startswith(phrase):
            return True
    return False


def _dedupe_items(
    items: list[dict],
    *,
    text_key: str = "text",
    limit: int = 5,
    min_items: int = 1,
    similarity_threshold: float = 0.82,
    banned_phrases: list[str] | None = None,
) -> list[dict]:
    banned_phrases = banned_phrases or []
    cleaned: list[dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        text = _clean_text(item.get(text_key))
        if not text:
            continue

        if _starts_with_banned_opener(text, banned_phrases):
            continue

        is_too_similar = False
        for existing in cleaned:
            if _text_similarity(text, existing.get(text_key, "")) >= similarity_threshold:
                is_too_similar = True
                break

        if is_too_similar:
            continue

        item[text_key] = text
        cleaned.append(item)

        if len(cleaned) >= limit:
            break

    if len(cleaned) < min_items:
        raise ValueError("Not enough valid items after diversity filtering.")

    return cleaned


def _label_from_score(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Decent"
    return "Weak"


def _score_hook_text(text: str) -> int:
    cleaned = _clean_text(text)
    words = [w for w in cleaned.split(" ") if w]
    word_count = len(words)
    char_count = len(cleaned)

    curiosity_words = {
        "why", "how", "secret", "mistake", "truth", "stop", "best", "worst",
        "nobody", "everyone", "before", "after", "never"
    }
    emotional_words = {
        "dangerous", "powerful", "simple", "shocking", "hard", "easy",
        "hidden", "real", "smart", "wrong"
    }

    brevity = 26 if 5 <= word_count <= 13 else 18 if 4 <= word_count <= 16 else 10
    readability = 24 if char_count <= 90 else 18 if char_count <= 120 else 12
    curiosity = 24 if any(w.lower().strip("?!.,:;") in curiosity_words for w in words) else 14
    punch = 24 if any(w.lower().strip("?!.,:;") in emotional_words for w in words) else 14

    return _clamp(brevity + readability + curiosity + punch, 35, 98)


def _score_title_text(text: str) -> int:
    cleaned = _clean_text(text)
    words = cleaned.split()
    score = 56

    if 35 <= len(cleaned) <= 65:
        score += 14
    elif 25 <= len(cleaned) <= 75:
        score += 8

    if 5 <= len(words) <= 11:
        score += 12
    elif 4 <= len(words) <= 13:
        score += 6

    if re.search(r"\b(how|why|what|mistake|truth|secret|best|worst)\b", cleaned.lower()):
        score += 10

    if any(mark in cleaned for mark in ["?", ":", "|"]):
        score += 6

    return _clamp(score, 35, 98)


def _score_caption_text(text: str) -> int:
    cleaned = _clean_text(text)
    score = 52

    if len(cleaned) >= 80:
        score += 8
    if len(cleaned) <= 260:
        score += 10
    if "\n" in text:
        score += 8
    if re.search(r"\b(comment|save|share|follow|watch|learn|try)\b", cleaned.lower()):
        score += 10
    if re.search(r"\b(you|your)\b", cleaned.lower()):
        score += 8

    return _clamp(score, 35, 96)


def _score_description_text(text: str) -> int:
    cleaned = _clean_text(text)
    score = 54

    if len(cleaned) >= 180:
        score += 10
    if len(cleaned) <= 1200:
        score += 8
    if any(token in text for token in ["\n", "-", "•"]):
        score += 8
    if re.search(r"\b(subscribe|comment|watch|learn|download|follow)\b", cleaned.lower()):
        score += 8
    if re.search(r"\b(how|why|what|guide|tips|mistakes|strategy)\b", cleaned.lower()):
        score += 8

    return _clamp(score, 35, 96)


def _score_script_text(text: str) -> int:
    cleaned = _clean_text(text)
    score = 58

    if 220 <= len(cleaned) <= 900:
        score += 12
    if re.search(r"\b(hook|cta|story|mistake|result|step)\b", cleaned.lower()):
        score += 8
    if text.count("\n") >= 3:
        score += 8
    if re.search(r"\b(you|your)\b", cleaned.lower()):
        score += 8

    return _clamp(score, 35, 97)


def _score_hashtag_set(tags: list[str]) -> int:
    score = 50
    if 6 <= len(tags) <= 12:
        score += 16
    elif len(tags) >= 4:
        score += 10

    lengths = {len(tag) for tag in tags}
    if len(lengths) >= 3:
        score += 8

    if len(set(tag.lower() for tag in tags)) == len(tags):
        score += 10

    return _clamp(score, 35, 96)


def _json_list_prompt(
    *,
    tool_name: str,
    context: Dict[str, Any],
    item_schema_hint: str,
    diversity_rules: list[str],
    count: int,
) -> str:
    joined_rules = "\n".join(f"- {rule}" for rule in diversity_rules)
    return (
        f"You are generating content for Hookora's {tool_name}.\n"
        "Return valid JSON only.\n"
        f"Return exactly this shape: {{\"items\": [{item_schema_hint}]}}\n\n"
        "Content context:\n"
        f"{_context_lines(context)}\n\n"
        "Diversity requirements:\n"
        f"{joined_rules}\n\n"
        f"Create {count} strong, distinct options."
    )


def _safe_items_from_response(data: Any) -> list[dict]:
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
        for value in data.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _fallback_titles(context: Dict[str, Any]) -> list[dict]:
    topic = context["topic"]
    kw = context["keywords"][0] if context["keywords"] else topic

    raw = [
        {"text": f"How to Win With {kw} Without Sounding Generic", "angle": "search intent"},
        {"text": f"The {topic} Mistake Most Creators Keep Repeating", "angle": "mistake-focused"},
        {"text": f"Why Your {topic} Strategy Still Feels the Same", "angle": "curiosity gap"},
        {"text": f"{topic}: The Smarter Way to Get Clicks", "angle": "outcome-driven"},
        {"text": f"Stop Using Weak {topic} Ideas. Try This Instead", "angle": "contrarian"},
        {"text": f"6 {topic} Angles That Actually Pull Attention", "angle": "framework/list"},
    ]

    items = []
    for item in raw:
        score = _score_title_text(item["text"])
        items.append({
            "text": item["text"],
            "angle": item["angle"],
            "score": score,
            "label": _label_from_score(score),
        })

    return _dedupe_items(
        items,
        limit=5,
        min_items=5,
        similarity_threshold=0.78,
        banned_phrases=context["banned_phrases"],
    )


def _fallback_hooks(context: Dict[str, Any], avoid_texts: list[str] | None = None) -> list[dict]:
    topic = context["topic"]
    pain_point = context["pain_point"] or "weak content performance"
    outcome = context["desired_outcome"] or "better results"
    avoid_texts = avoid_texts or []

    raw = [
        {"text": f"Why does your {topic} still sound like everyone else?", "angle": "question"},
        {"text": f"Most people are making {topic} harder than it needs to be.", "angle": "contrarian"},
        {"text": f"The fastest way to kill {topic} is to ignore this mistake.", "angle": "mistake"},
        {"text": f"We changed one thing in our {topic} process and got {outcome}.", "angle": "proof/result"},
        {"text": f"I realized our {topic} was weak the moment this happened.", "angle": "story"},
        {"text": f"If your audience struggles with {pain_point}, stop opening like this.", "angle": "warning"},
    ]

    items = []
    for item in raw:
        if any(_text_similarity(item["text"], old) >= 0.78 for old in avoid_texts):
            continue
        score = _score_hook_text(item["text"])
        items.append({
            "text": item["text"],
            "angle": item["angle"],
            "score": score,
            "label": _label_from_score(score),
            "platform": context["platform"],
        })

    return _dedupe_items(
        items,
        limit=6,
        min_items=3,
        similarity_threshold=0.78,
        banned_phrases=context["banned_phrases"],
    )


def _fallback_captions(context: Dict[str, Any]) -> list[dict]:
    topic = context["topic"]
    cta = context["call_to_action"] or "Comment if you want part 2."
    platform = context["platform"]

    raw = [
        {
            "text": (
                f"Most {topic} content sounds polished but forgettable.\n\n"
                f"The real difference is not the topic. It is the angle, tension, and clarity behind it.\n\n"
                f"{cta}"
            ),
            "angle": "insight-driven",
        },
        {
            "text": (
                f"I used to think better {topic} meant adding more words.\n"
                f"It usually means cutting the weak parts and making the core idea sharper.\n\n"
                f"Would you post this on {platform}?"
            ),
            "angle": "story-led",
        },
        {
            "text": (
                f"If your {topic} keeps blending in, your audience is not the problem.\n"
                f"Your positioning probably is.\n\n"
                f"Save this before your next post."
            ),
            "angle": "cta-heavy",
        },
        {
            "text": (
                f"Quick reminder: strong {topic} content does not start with more effort.\n"
                f"It starts with a clearer promise.\n\n"
                f"What is the first thing you would improve?"
            ),
            "angle": "conversation starter",
        },
    ]

    items = []
    for item in raw:
        score = _score_caption_text(item["text"])
        items.append({
            "text": item["text"],
            "angle": item["angle"],
            "score": score,
            "label": _label_from_score(score),
        })

    return _dedupe_items(
        items,
        limit=4,
        min_items=4,
        similarity_threshold=0.78,
        banned_phrases=context["banned_phrases"],
    )


def _fallback_description_rewrites(context: Dict[str, Any]) -> list[dict]:
    topic = context["topic"]
    kw = ", ".join(context["keywords"][:3]) if context["keywords"] else topic
    cta = context["call_to_action"] or "Subscribe for more."
    platform = context["platform"]

    raw_texts = [
        (
            f"If you want stronger {topic} results on {platform}, this breakdown shows what actually matters.\n\n"
            f"In this content, you will learn:\n"
            f"- why weak structure hurts performance\n"
            f"- how to sharpen your core message\n"
            f"- what makes people stop, click, and keep watching\n\n"
            f"Keywords: {kw}\n\n"
            f"{cta}"
        ),
        (
            f"Most creators do not have a {topic} problem. They have a clarity problem.\n\n"
            f"This piece breaks down how to improve positioning, message strength, and attention flow so your content feels sharper from the first line.\n\n"
            f"Ideal for creators working on {kw}.\n\n"
            f"{cta}"
        ),
        (
            f"Trying to improve your {topic}? This guide gives you a more strategic way to think about hooks, titles, captions, and audience relevance.\n\n"
            f"Watch or read to understand what makes content stronger, more searchable, and more worth clicking.\n\n"
            f"{cta}"
        ),
    ]

    items = []
    for idx, text in enumerate(raw_texts):
        score = _score_description_text(text)
        items.append({
            "text": text,
            "angle": DESCRIPTION_ARCHETYPES[idx],
            "score": score,
            "label": _label_from_score(score),
        })

    return _dedupe_items(
        items,
        limit=3,
        min_items=3,
        similarity_threshold=0.76,
        banned_phrases=context["banned_phrases"],
    )


def _fallback_scripts(context: Dict[str, Any]) -> list[dict]:
    topic = context["topic"]
    cta = context["call_to_action"] or "Follow for more."
    outcome = context["desired_outcome"] or "better performance"

    raw = [
        {
            "title": "Direct and Punchy",
            "angle": "direct-to-camera",
            "text": (
                f"Hook:\nYour {topic} feels weak because it sounds predictable.\n\n"
                f"Body:\nMost people think improvement means adding more. It usually means tightening the promise, cutting filler, and leading with a stronger tension point.\n"
                f"If your audience cannot feel the value fast, they scroll.\n\n"
                f"CTA:\nUse one sharper angle today and test the difference. {cta}"
            ),
        },
        {
            "title": "Story-Led Version",
            "angle": "story-led",
            "text": (
                f"Hook:\nI realized our {topic} had a problem when the idea was good but nobody cared.\n\n"
                f"Body:\nThe issue was not effort. The issue was framing. We were saying what it was instead of why it mattered.\n"
                f"Once we changed the angle, the message felt clearer and the result was {outcome}.\n\n"
                f"CTA:\nLook at your current opener and rewrite it with more tension. {cta}"
            ),
        },
        {
            "title": "Contrarian Explainer",
            "angle": "contrarian explainer",
            "text": (
                f"Hook:\nBetter {topic} does not start with better design. It starts with better positioning.\n\n"
                f"Body:\nIf the promise is weak, the content feels weak.\n"
                f"Strong creators know how to make the audience feel the gap between where they are and where they want to be.\n"
                f"That gap creates attention.\n\n"
                f"CTA:\nTry that on your next post and tell me what changes. {cta}"
            ),
        },
        {
            "title": "Problem-Solution Flow",
            "angle": "problem-solution",
            "text": (
                f"Hook:\nHere is why your {topic} still blends in.\n\n"
                f"Body:\nProblem: the structure is familiar and the promise is vague.\n"
                f"Solution: pick one audience pain point, one clear outcome, and one memorable angle.\n"
                f"That alone makes the content feel more specific and more clickable.\n\n"
                f"CTA:\nSave this and use it before your next upload. {cta}"
            ),
        },
    ]

    items = []
    for item in raw:
        score = _score_script_text(item["text"])
        items.append({
            "title": item["title"],
            "angle": item["angle"],
            "text": item["text"],
            "score": score,
            "label": _label_from_score(score),
        })

    return _dedupe_items(
        items,
        text_key="text",
        limit=4,
        min_items=4,
        similarity_threshold=0.76,
        banned_phrases=context["banned_phrases"],
    )


def _fallback_hashtag_sets(context: Dict[str, Any]) -> list[dict]:
    topic_slug = re.sub(r"[^a-z0-9]+", "", context["topic"].lower()) or "content"
    keywords = [re.sub(r"[^a-z0-9]+", "", kw.lower()) for kw in context["keywords"]]
    keywords = [kw for kw in keywords if kw]
    platform_slug = re.sub(r"[^a-z0-9]+", "", context["platform"].lower())

    raw_sets = [
        {
            "title": "Discovery Set",
            "description": "Broader hashtags for wider reach.",
            "tags": [
                f"#{topic_slug}",
                "#contentstrategy",
                "#socialmedia",
                "#creatorgrowth",
                "#digitalmarketing",
                f"#{platform_slug or 'contenttips'}",
            ] + [f"#{kw}" for kw in keywords[:2]],
            "angle": "discovery",
        },
        {
            "title": "Balanced Set",
            "description": "Mix of broad and specific hashtags.",
            "tags": [
                f"#{topic_slug}tips",
                "#audiencegrowth",
                "#contenthooks",
                "#contentwriting",
                "#marketingideas",
                "#creatorbrand",
            ] + [f"#{kw}" for kw in keywords[:2]],
            "angle": "balanced",
        },
        {
            "title": "Niche Set",
            "description": "Tighter targeting for more qualified discovery.",
            "tags": [
                f"#{topic_slug}strategy",
                f"#{topic_slug}content",
                "#hookwriting",
                "#contentpositioning",
                "#conversioncontent",
                "#brandmessaging",
            ] + [f"#{kw}strategy" for kw in keywords[:2]],
            "angle": "niche",
        },
    ]

    items = []
    for item in raw_sets:
        tags = []
        seen = set()
        for tag in item["tags"]:
            normalized = f"#{re.sub(r'[^a-zA-Z0-9_]+', '', tag.replace('#', ''))}"
            if normalized == "#":
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            tags.append(normalized)

        tags = tags[:10]
        score = _score_hashtag_set(tags)
        items.append({
            "title": item["title"],
            "description": item["description"],
            "tags": tags,
            "angle": item["angle"],
            "score": score,
            "label": _label_from_score(score),
        })

    return items


def _fallback_thumbnail_analysis(context: Dict[str, Any]) -> dict:
    thumbnail_text = context["thumbnail_text"] or context["topic"]
    cleaned = _clean_text(thumbnail_text)
    words = cleaned.split()

    score = 56
    feedback = []
    if len(words) <= 4:
        score += 14
        feedback.append("The text is short enough for a thumbnail.")
    else:
        feedback.append("The text is a bit long for a high-impact thumbnail.")

    if re.search(r"\b(why|how|secret|truth|mistake|worst|best)\b", cleaned.lower()):
        score += 10
        feedback.append("It includes a curiosity or tension word.")
    else:
        feedback.append("It could use a stronger curiosity or tension word.")

    if len(cleaned) <= 28:
        score += 10

    score = _clamp(score, 35, 95)
    suggestions = [
        {"text": "The Real Reason It Failed", "angle": "curiosity"},
        {"text": "Stop Doing This", "angle": "warning"},
        {"text": "What Changed Everything", "angle": "story"},
        {"text": "Most Creators Miss This", "angle": "mistake"},
    ]

    return {
        "text": cleaned,
        "score": score,
        "label": _label_from_score(score),
        "feedback": feedback,
        "metrics": {
            "word_count": len(words),
            "character_count": len(cleaned),
            "clarity": _clamp(score - 4, 30, 95),
            "curiosity": _clamp(score + 2, 30, 95),
            "brevity": _clamp(88 if len(words) <= 4 else 62, 30, 95),
        },
        "suggestions": suggestions,
    }


def generate_titles(
    topic: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
) -> list[dict]:
    context = _build_context_dict(
        topic=topic,
        platform="YouTube",
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
    )

    instructions = (
        "You are a world-class YouTube title strategist.\n"
        "Write titles that are distinct from one another, not minor variations.\n"
        "Keep them specific, clickable, SEO-aware, and natural.\n"
        "Do not use generic filler or repetitive structures."
    )

    prompt = _json_list_prompt(
        tool_name="YouTube Title Optimizer",
        context=context,
        item_schema_hint='{"text":"title text","angle":"one of the requested archetypes"}',
        diversity_rules=[
            f"Use six different archetypes across the raw candidates: {', '.join(TITLE_ARCHETYPES)}.",
            "Avoid repeating the same opener or sentence structure.",
            "Blend curiosity with search intent where appropriate.",
            "Prefer clear benefit, tension, or mistake language over vague hype.",
        ],
        count=6,
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=1400)
        raw_items = _safe_items_from_response(data)

        items: list[dict] = []
        for item in raw_items:
            text = _clean_text(item.get("text"))
            if not text:
                continue
            score = _score_title_text(text)
            items.append({
                "text": text,
                "angle": _clean_text(item.get("angle")) or "general",
                "score": score,
                "label": _label_from_score(score),
            })

        return _dedupe_items(
            items,
            limit=5,
            min_items=5,
            similarity_threshold=0.78,
            banned_phrases=context["banned_phrases"],
        )
    except Exception:
        return _fallback_titles(context)


def generate_hooks(
    *,
    topic: str,
    platform: str | None = None,
    template: str | None = "curiosity",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
    avoid_texts: Any = None,
) -> list[dict]:
    normalized_template = _clean_text(template or "curiosity").lower()
    template_guidance = HOOK_TEMPLATE_GUIDANCE.get(
        normalized_template,
        HOOK_TEMPLATE_GUIDANCE["curiosity"]
    )

    context = _build_context_dict(
        topic=topic,
        platform=platform,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
        template=normalized_template,
    )

    avoid_items = _normalize_avoid_list(avoid_texts) + context["examples_to_avoid"]

    instructions = (
        "You are an elite short-form hook writer.\n"
        "Generate hooks that feel original, pattern-interrupting, and high-signal.\n"
        "Do not return small rewrites of the same line.\n"
        "Avoid generic creator-cliche language."
    )

    prompt = (
        _json_list_prompt(
            tool_name="Viral Hook Generator",
            context=context,
            item_schema_hint='{"text":"hook text","angle":"question|contrarian|mistake|proof/result|story|warning"}',
            diversity_rules=[
                f"Primary style guidance: {template_guidance}",
                f"Cover these six hook families across raw candidates: {', '.join(HOOK_ARCHETYPES)}.",
                "Every line must feel like a different angle, not a paraphrase.",
                "Do not start with generic filler like 'here are', 'in this video', or 'let's talk about'.",
                "Avoid similarity to the provided examples to avoid and avoid_texts list.",
            ],
            count=8,
        )
        + "\n\nAvoid lines similar to these prior outputs:\n"
        + ("\n".join(f"- {item}" for item in avoid_items) if avoid_items else "- None provided")
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=1600)
        raw_items = _safe_items_from_response(data)

        items: list[dict] = []
        for item in raw_items:
            text = _clean_text(item.get("text"))
            if not text:
                continue

            if any(_text_similarity(text, old) >= 0.78 for old in avoid_items):
                continue

            score = _score_hook_text(text)
            items.append({
                "text": text,
                "angle": _clean_text(item.get("angle")) or "general",
                "score": score,
                "label": _label_from_score(score),
                "platform": context["platform"],
            })

        return _dedupe_items(
            items,
            limit=6,
            min_items=4,
            similarity_threshold=0.78,
            banned_phrases=context["banned_phrases"],
        )
    except Exception:
        return _fallback_hooks(context, avoid_items)


def generate_single_hook(
    *,
    topic: str,
    platform: str | None = None,
    template: str | None = "curiosity",
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
    avoid_texts: Any = None,
) -> dict:
    hooks = generate_hooks(
        topic=topic,
        platform=platform,
        template=template,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
        avoid_texts=avoid_texts,
    )
    return hooks[0]


def generate_captions(
    topic: str,
    platform: str,
    tone: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
) -> list[dict]:
    context = _build_context_dict(
        topic=topic,
        platform=platform,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
        tone=tone,
    )

    instructions = (
        "You are a high-performing social media caption strategist.\n"
        "Write captions that feel native to the platform, varied in structure, and non-repetitive.\n"
        "Use clean formatting and natural language."
    )

    prompt = _json_list_prompt(
        tool_name="Caption Generator",
        context=context,
        item_schema_hint='{"text":"caption text","angle":"conversation starter|insight-driven|story-led|cta-heavy"}',
        diversity_rules=[
            f"Use these four caption styles across the raw set: {', '.join(CAPTION_ARCHETYPES)}.",
            "At least one caption should be short and punchy, and at least one should be more developed.",
            "Do not use the same CTA wording in every option.",
            "Make the voice match the requested tone and platform.",
        ],
        count=5,
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=1800)
        raw_items = _safe_items_from_response(data)

        items = []
        for item in raw_items:
            text = _clean_text(item.get("text"))
            if not text:
                continue
            score = _score_caption_text(text)
            items.append({
                "text": text,
                "angle": _clean_text(item.get("angle")) or "general",
                "score": score,
                "label": _label_from_score(score),
            })

        return _dedupe_items(
            items,
            limit=4,
            min_items=4,
            similarity_threshold=0.78,
            banned_phrases=context["banned_phrases"],
        )
    except Exception:
        return _fallback_captions(context)


def generate_hashtags(
    topic: str,
    platform: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
) -> list[dict]:
    context = _build_context_dict(
        topic=topic,
        platform=platform,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
    )

    instructions = (
        "You are a social discovery strategist.\n"
        "Create three distinct hashtag sets: discovery, balanced, and niche.\n"
        "Avoid duplicated tags and avoid weak generic filler unless it clearly helps relevance."
    )

    prompt = _json_list_prompt(
        tool_name="Hashtag Generator",
        context=context,
        item_schema_hint='{"title":"set name","description":"what this set is for","tags":["#one","#two"],"angle":"discovery|balanced|niche"}',
        diversity_rules=[
            "Return three sets only.",
            "Each set must have a different reach strategy.",
            "Each tag must start with # and contain no spaces.",
            "Keep each set tight and relevant, not bloated.",
        ],
        count=3,
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=1500)
        raw_items = _safe_items_from_response(data)

        items = []
        for item in raw_items:
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                continue

            cleaned_tags = []
            seen = set()
            for tag in tags:
                normalized = _clean_text(str(tag))
                if not normalized:
                    continue
                normalized = f"#{re.sub(r'[^a-zA-Z0-9_]+', '', normalized.replace('#', ''))}"
                if normalized == "#":
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned_tags.append(normalized)

            if len(cleaned_tags) < 4:
                continue

            cleaned_tags = cleaned_tags[:10]
            score = _score_hashtag_set(cleaned_tags)
            items.append({
                "title": _clean_text(item.get("title")) or "Hashtag Set",
                "description": _clean_text(item.get("description")) or "Relevant hashtag cluster.",
                "tags": cleaned_tags,
                "angle": _clean_text(item.get("angle")) or "general",
                "score": score,
                "label": _label_from_score(score),
            })

        if len(items) >= 3:
            return items[:3]
        return _fallback_hashtag_sets(context)
    except Exception:
        return _fallback_hashtag_sets(context)


def generate_description_rewrites(
    topic: str,
    platform: str,
    original_description: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
) -> list[dict]:
    context = _build_context_dict(
        topic=topic,
        platform=platform,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
        original_description=original_description,
    )

    instructions = (
        "You are an SEO-aware content description strategist.\n"
        "Rewrite descriptions with stronger clarity, better scanning, and stronger value communication.\n"
        "Make the versions meaningfully different, not lightly reworded."
    )

    prompt = _json_list_prompt(
        tool_name="Description Rewriter",
        context=context,
        item_schema_hint='{"text":"rewritten description","angle":"seo structured|benefit-first|authority-led"}',
        diversity_rules=[
            f"Use these three styles across the final set: {', '.join(DESCRIPTION_ARCHETYPES)}.",
            "At least one version should use bullets or clear scannable structure.",
            "If keywords are provided, weave them in naturally without stuffing.",
            "Do not output shallow generic intros.",
        ],
        count=4,
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=2200)
        raw_items = _safe_items_from_response(data)

        items = []
        for item in raw_items:
            text = _clean_text(item.get("text"))
            if not text:
                continue
            score = _score_description_text(text)
            items.append({
                "text": text,
                "angle": _clean_text(item.get("angle")) or "general",
                "score": score,
                "label": _label_from_score(score),
            })

        return _dedupe_items(
            items,
            limit=3,
            min_items=3,
            similarity_threshold=0.76,
            banned_phrases=context["banned_phrases"],
        )
    except Exception:
        return _fallback_description_rewrites(context)


def generate_scripts(
    topic: str,
    platform: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    product_or_offer: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    call_to_action: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    content_type: str | None = None,
    extra_context: str | None = None,
    examples_to_avoid: Any = None,
    banned_phrases: Any = None,
) -> list[dict]:
    context = _build_context_dict(
        topic=topic,
        platform=platform,
        language=language,
        audience=audience,
        goal=goal,
        product_or_offer=product_or_offer,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        call_to_action=call_to_action,
        keywords=keywords,
        brand_voice=brand_voice,
        content_type=content_type,
        extra_context=extra_context,
        examples_to_avoid=examples_to_avoid,
        banned_phrases=banned_phrases,
    )

    instructions = (
        "You are a short-form script strategist.\n"
        "Generate distinct script directions, each with a different structure and narrative logic.\n"
        "Avoid bland templates and repeated opening patterns."
    )

    prompt = _json_list_prompt(
        tool_name="Short Script Generator",
        context=context,
        item_schema_hint='{"title":"script title","text":"full script with clear sections","angle":"direct-to-camera|story-led|contrarian explainer|problem-solution"}',
        diversity_rules=[
            f"Use these four structures across the final set: {', '.join(SCRIPT_ARCHETYPES)}.",
            "Each script should feel like a different creative path.",
            "Make the first line strong enough to stop the scroll.",
            "Include a CTA, but vary how the CTA is phrased.",
        ],
        count=5,
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=2600)
        raw_items = _safe_items_from_response(data)

        items = []
        for item in raw_items:
            text = _clean_text(item.get("text"))
            if not text:
                continue
            score = _score_script_text(text)
            items.append({
                "title": _clean_text(item.get("title")) or "Script Option",
                "angle": _clean_text(item.get("angle")) or "general",
                "text": text,
                "score": score,
                "label": _label_from_score(score),
            })

        return _dedupe_items(
            items,
            text_key="text",
            limit=4,
            min_items=4,
            similarity_threshold=0.76,
            banned_phrases=context["banned_phrases"],
        )
    except Exception:
        return _fallback_scripts(context)


def analyze_thumbnail_text(
    topic: str,
    thumbnail_text: str,
    *,
    language: str = "en",
    audience: str | None = None,
    goal: str | None = None,
    pain_point: str | None = None,
    desired_outcome: str | None = None,
    angle: str | None = None,
    keywords: Any = None,
    brand_voice: str | None = None,
    extra_context: str | None = None,
    banned_phrases: Any = None,
) -> dict:
    context = _build_context_dict(
        topic=topic,
        platform="YouTube",
        language=language,
        audience=audience,
        goal=goal,
        pain_point=pain_point,
        desired_outcome=desired_outcome,
        angle=angle,
        keywords=keywords,
        brand_voice=brand_voice,
        extra_context=extra_context,
        banned_phrases=banned_phrases,
        thumbnail_text=thumbnail_text,
    )

    instructions = (
        "You are a thumbnail text analyst for YouTube.\n"
        "Score the thumbnail text for brevity, clarity, tension, and clickability.\n"
        "Return a grounded analysis and better replacement suggestions."
    )

    prompt = (
        "Return valid JSON only with this shape:\n"
        "{"
        "\"analysis\":{"
        "\"text\":\"thumbnail text\","
        "\"score\":0,"
        "\"label\":\"Strong\","
        "\"feedback\":[\"...\"],"
        "\"metrics\":{\"word_count\":0,\"character_count\":0,\"clarity\":0,\"curiosity\":0,\"brevity\":0}"
        "},"
        "\"suggestions\":[{\"text\":\"replacement text\",\"angle\":\"curiosity\"}]"
        "}\n\n"
        "Context:\n"
        f"{_context_lines(context)}\n\n"
        "Requirements:\n"
        "- Keep suggestions short enough for a thumbnail.\n"
        "- Make suggestions distinct from one another.\n"
        "- Prefer 2 to 5 words for suggestions.\n"
        "- Mention if the current text is too vague, too long, or too safe."
    )

    try:
        data = _call_json_model(instructions=instructions, prompt=prompt, max_output_tokens=1400)
        analysis = data.get("analysis") if isinstance(data, dict) else None
        suggestions = data.get("suggestions") if isinstance(data, dict) else None

        if not isinstance(analysis, dict):
            raise ValueError("Missing analysis.")

        cleaned_text = _clean_text(analysis.get("text") or thumbnail_text or topic)
        score = _clamp(analysis.get("score", 70), 30, 98)
        feedback = analysis.get("feedback") if isinstance(analysis.get("feedback"), list) else []
        metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else {}

        cleaned_suggestions = []
        if isinstance(suggestions, list):
            for item in suggestions:
                if not isinstance(item, dict):
                    continue
                text = _clean_text(item.get("text"))
                if not text:
                    continue
                cleaned_suggestions.append({
                    "text": text,
                    "angle": _clean_text(item.get("angle")) or "general",
                })

        cleaned_suggestions = _dedupe_items(
            cleaned_suggestions,
            limit=4,
            min_items=2,
            similarity_threshold=0.75,
            banned_phrases=context["banned_phrases"],
        )

        return {
            "text": cleaned_text,
            "score": score,
            "label": _label_from_score(score),
            "feedback": [_clean_text(str(item)) for item in feedback if _clean_text(str(item))],
            "metrics": {
                "word_count": int(metrics.get("word_count", len(cleaned_text.split()))),
                "character_count": int(metrics.get("character_count", len(cleaned_text))),
                "clarity": _clamp(metrics.get("clarity", score), 30, 98),
                "curiosity": _clamp(metrics.get("curiosity", score), 30, 98),
                "brevity": _clamp(metrics.get("brevity", 90 if len(cleaned_text.split()) <= 4 else 62), 30, 98),
            },
            "suggestions": cleaned_suggestions,
        }
    except Exception:
        return _fallback_thumbnail_analysis(context)