import json
import re
from typing import Any

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


HOOK_TEMPLATE_GUIDANCE = {
    "curiosity": "Focus on curiosity, intrigue, and making the audience want the next line.",
    "authority": "Focus on confidence, expertise, and strong insight.",
    "problem": "Focus on pain points, mistakes, frustration, and urgent relevance.",
    "benefit": "Focus on transformation, clear value, and results.",
    "controversy": "Focus on tension, bold perspective, and pattern interruption.",
    "story": "Focus on narrative setup, emotional tension, and human relatability.",
}


def _ensure_openai_ready() -> None:
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is missing. Add OPENAI_API_KEY to your backend environment."
        )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


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
    max_output_tokens: int = 900,
    temperature: float = 0.8,
) -> Any:
    _ensure_openai_ready()

    try:
        response = client.responses.create(
            model=settings.openai_model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
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


def _normalize_string_list(value: Any, *, limit: int, min_items: int = 1) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list.")

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            continue
        text = _clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break

    if len(cleaned) < min_items:
        raise ValueError("Not enough valid items returned.")

    return cleaned


def _normalize_platform(platform: str | None) -> str:
    cleaned = _clean_text(platform or "")
    if not cleaned:
        return "General"

    return SUPPORTED_HOOK_PLATFORMS.get(cleaned.lower(), cleaned)


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

    brevity = 28 if 6 <= word_count <= 14 else 20 if 4 <= word_count <= 16 else 12
    readability = 24 if char_count <= 90 else 18 if char_count <= 120 else 12
    curiosity = 24 if any(w.lower().strip("?!.,:;") in curiosity_words for w in words) else 14
    punch = 24 if any(w.lower().strip("?!.,:;") in emotional_words for w in words) else 14

    return _clamp(brevity + readability + curiosity + punch, 35, 98)


def _fallback_titles(topic: str) -> list[str]:
    topic = _clean_text(topic)
    return [
        f"{topic}: What Most Creators Get Wrong",
        f"How to Make {topic} More Clickable",
        f"{topic} Explained in a Smarter Way",
        f"The Better Way to Approach {topic}",
        f"{topic}: Simple Changes That Make a Big Difference",
    ]


def _fallback_captions(topic: str, platform: str, tone: str) -> list[str]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)
    tone = _clean_text(tone)
    return [
        f"{topic} matters more than most people think. Here’s a {tone} take for {platform}.",
        f"If you’re working on {topic}, this is your reminder to keep it clear, sharp, and consistent.",
        f"{topic} can change how your content lands. Small improvements make a big difference.",
        f"Trying to improve your {platform} content? Start by making {topic} stronger and easier to understand.",
    ]


def _fallback_hashtags(topic: str, platform: str) -> list[list[str]]:
    slug = re.sub(r"[^a-zA-Z0-9 ]+", "", topic).strip().replace(" ", "")
    platform_slug = re.sub(r"[^a-zA-Z0-9]+", "", platform)
    base = [
        f"#{slug}" if slug else "#content",
        f"#{platform_slug}" if platform_slug else "#socialmedia",
        "#creator",
        "#contentcreation",
        "#growth",
    ]
    return [
        base,
        base + ["#marketing", "#branding"],
        base + ["#audience", "#strategy"],
    ]


def _fallback_rewrites(topic: str, platform: str, original_description: str) -> list[str]:
    base = _clean_text(original_description) or f"{topic} for {platform}"
    return [
        f"{base} — rewritten for clearer value, stronger positioning, and a more engaging tone.",
        f"A sharper version of {base} that feels more intentional and easier to understand.",
        f"{base} presented in a cleaner way so the message lands faster and more clearly.",
    ]


def _fallback_scripts(topic: str, platform: str) -> list[str]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)
    return [
        f"Hook: Most people approach {topic} the wrong way.\n\nBody: If you want better results on {platform}, start by making your message clearer and easier to follow.\n\nCTA: Follow for more creator workflow tips.",
        f"Hook: Here’s how to make {topic} stronger.\n\nBody: Focus on clarity first, then improve the wording so your audience understands the value immediately.\n\nCTA: Save this if you want better {platform} content.",
    ]


def _fallback_hook_texts(topic: str, platform: str, template: str) -> list[str]:
    topic = _clean_text(topic)
    platform = _normalize_platform(platform)
    template = _clean_text(template or "curiosity").lower()

    if template == "problem":
        return [
            f"Most people keep ruining their {topic} without realizing it.",
            f"If your {platform} content on {topic} is not landing, this is probably why.",
            f"The biggest mistake creators make with {topic} is simpler than you think.",
            f"Stop doing this if you want better results with {topic}.",
            f"This one mistake is quietly killing your {topic} content.",
        ]

    if template == "benefit":
        return [
            f"How to make {topic} instantly more compelling.",
            f"The simple shift that makes {topic} perform better.",
            f"How creators make {topic} look effortless.",
            f"The fastest way to improve your {topic} content.",
            f"What better {topic} looks like in real content.",
        ]

    if template == "authority":
        return [
            f"Here’s what strong creators understand about {topic}.",
            f"The smartest way to approach {topic} right now.",
            f"What actually works when it comes to {topic}.",
            f"If you want better {topic}, start here.",
            f"This is the standard your {topic} content should reach.",
        ]

    if template == "story":
        return [
            f"I thought I understood {topic} until this changed everything.",
            f"At first, {topic} seemed simple. It wasn’t.",
            f"The moment I realized my approach to {topic} was wrong.",
            f"This changed how I think about {topic}.",
            f"I learned this lesson about {topic} the hard way.",
        ]

    if template == "controversy":
        return [
            f"Most advice about {topic} is completely off.",
            f"Why I disagree with how people talk about {topic}.",
            f"The truth about {topic} is less comfortable than people admit.",
            f"Everyone says this about {topic}. They’re wrong.",
            f"The popular approach to {topic} is hurting creators.",
        ]

    return [
        f"Why nobody talks about {topic} the right way.",
        f"The real reason your {topic} content is not connecting.",
        f"What makes {topic} instantly more interesting?",
        f"Most creators miss this about {topic}.",
        f"Before you post about {topic}, read this.",
    ]


def _score_thumbnail_text(text: str) -> dict[str, Any]:
    cleaned = _clean_text(text)
    words = [w for w in cleaned.split(" ") if w]
    word_count = len(words)
    char_count = len(cleaned)

    brevity_score = 30 if 2 <= word_count <= 5 else 20 if 1 <= word_count <= 7 else 10
    clarity_score = 25 if char_count <= 32 else 18 if char_count <= 45 else 10
    emphasis_score = 20 if any(word.isupper() for word in words) else 12
    curiosity_words = {"why", "how", "secret", "mistake", "truth", "stop", "best", "worst"}
    curiosity_score = 25 if any(w.lower().strip("?!.,") in curiosity_words for w in words) else 14

    total = _clamp(brevity_score + clarity_score + emphasis_score + curiosity_score, 0, 100)

    if total >= 80:
        label = "Strong"
    elif total >= 60:
        label = "Decent"
    else:
        label = "Weak"

    feedback: list[str] = []
    if word_count > 6:
        feedback.append("Shorter thumbnail text usually reads faster.")
    if char_count > 40:
        feedback.append("Try reducing the character count for faster visual scanning.")
    if not any(w.isupper() for w in words):
        feedback.append("Consider emphasizing one important word.")
    if not any(w.lower().strip('?!.,') in curiosity_words for w in words):
        feedback.append("A stronger tension or curiosity word could help.")

    if not feedback:
        feedback.append("The text is compact, readable, and has good attention potential.")

    return {
        "score": total,
        "label": label,
        "feedback": feedback,
        "metrics": {
            "word_count": word_count,
            "character_count": char_count,
        },
    }


def generate_hooks(topic: str, platform: str | None = None, template: str | None = "curiosity") -> list[dict[str, Any]]:
    topic = _clean_text(topic)
    platform = _normalize_platform(platform)
    template = _clean_text(template or "curiosity").lower()

    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")

    template_guidance = HOOK_TEMPLATE_GUIDANCE.get(template, HOOK_TEMPLATE_GUIDANCE["curiosity"])

    instructions = (
        "You are a viral hook generator for creators. "
        "Return valid JSON with a single key named hooks. "
        "The value must be an array of exactly 5 hook strings. "
        "Hooks should be concise, punchy, curiosity-driven, platform-aware, and not spammy. "
        "Do not include numbering, markdown, or explanations."
    )

    prompt = (
        f"Topic: {topic}\n"
        f"Platform: {platform}\n"
        f"Template: {template}\n"
        f"Guidance: {template_guidance}"
    )

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=700,
            temperature=0.95,
        )
        hook_texts = _normalize_string_list(result.get("hooks"), limit=5, min_items=3)
    except Exception:
        hook_texts = _fallback_hook_texts(topic, platform, template)

    return [
        {
            "text": text,
            "score": _score_hook_text(text),
            "platform": platform,
        }
        for text in hook_texts[:5]
    ]


def generate_single_hook(topic: str, platform: str | None = None, template: str | None = "curiosity") -> dict[str, Any]:
    hooks = generate_hooks(topic=topic, platform=platform, template=template)
    if not hooks:
        raise HTTPException(status_code=500, detail="Failed to generate hook.")
    return hooks[0]


def generate_titles(topic: str) -> list[str]:
    topic = _clean_text(topic)
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")

    instructions = (
        "You are an expert YouTube title optimizer for creators. "
        "Return valid JSON with a single key named titles. "
        "The value must be an array of exactly 5 distinct YouTube title strings. "
        "Titles should be clear, clickable, curiosity-driven, and not spammy. "
        "Do not include numbering, markdown, or explanations."
    )

    prompt = f"Topic: {topic}"

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=500,
            temperature=0.9,
        )
        return _normalize_string_list(result.get("titles"), limit=5, min_items=3)
    except Exception:
        return _fallback_titles(topic)


def generate_captions(topic: str, platform: str, tone: str = "engaging") -> list[str]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)
    tone = _clean_text(tone) or "engaging"

    if not topic or not platform:
        raise HTTPException(status_code=400, detail="Topic and platform are required.")

    instructions = (
        "You are a social media caption writer for creators. "
        "Return valid JSON with a single key named captions. "
        "The value must be an array of exactly 4 caption strings. "
        "Captions should fit the requested platform and tone, feel natural, and be ready to post. "
        "Do not include hashtags unless naturally needed. Do not include explanations."
    )

    prompt = f"Topic: {topic}\nPlatform: {platform}\nTone: {tone}"

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=700,
            temperature=0.9,
        )
        return _normalize_string_list(result.get("captions"), limit=4, min_items=3)
    except Exception:
        return _fallback_captions(topic, platform, tone)


def generate_hashtags(topic: str, platform: str) -> list[list[str]]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)

    if not topic or not platform:
        raise HTTPException(status_code=400, detail="Topic and platform are required.")

    instructions = (
        "You are a hashtag generator for creators. "
        "Return valid JSON with one key named sets. "
        "The value must be an array containing exactly 3 hashtag arrays. "
        "Each hashtag array should contain 5 to 8 relevant hashtags as strings starting with #. "
        "Do not include explanations."
    )

    prompt = f"Topic: {topic}\nPlatform: {platform}"

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=600,
            temperature=0.8,
        )
        raw_sets = result.get("sets")
        if not isinstance(raw_sets, list) or not raw_sets:
            raise ValueError("Invalid hashtag sets.")

        cleaned_sets: list[list[str]] = []
        for item in raw_sets[:3]:
            cleaned = _normalize_string_list(item, limit=8, min_items=3)
            normalized = []
            for tag in cleaned:
                tag_text = tag if tag.startswith("#") else f"#{tag.lstrip('#')}"
                normalized.append(tag_text.replace(" ", ""))
            cleaned_sets.append(normalized)

        if not cleaned_sets:
            raise ValueError("No valid hashtag sets.")
        return cleaned_sets
    except Exception:
        return _fallback_hashtags(topic, platform)


def generate_description_rewrites(topic: str, platform: str, original_description: str = "") -> list[str]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)
    original_description = (original_description or "").strip()

    if not topic or not platform:
        raise HTTPException(status_code=400, detail="Topic and platform are required.")

    instructions = (
        "You are a content description rewriter for creators. "
        "Return valid JSON with one key named rewrites. "
        "The value must be an array of exactly 3 rewritten description strings. "
        "The rewrites should be clearer, stronger, and more intentional than the original. "
        "Do not include explanations."
    )

    prompt = (
        f"Topic: {topic}\n"
        f"Platform: {platform}\n"
        f"Original description: {original_description or 'None provided. Create from the topic.'}"
    )

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=900,
            temperature=0.8,
        )
        return _normalize_string_list(result.get("rewrites"), limit=3, min_items=2)
    except Exception:
        return _fallback_rewrites(topic, platform, original_description)


def generate_scripts(topic: str, platform: str) -> list[str]:
    topic = _clean_text(topic)
    platform = _clean_text(platform)

    if not topic or not platform:
        raise HTTPException(status_code=400, detail="Topic and platform are required.")

    instructions = (
        "You are a short-form script writer for creators. "
        "Return valid JSON with one key named scripts. "
        "The value must be an array of exactly 2 short script strings. "
        "Each script should include a hook, body, and CTA in plain text. "
        "Do not include explanations outside the scripts."
    )

    prompt = f"Topic: {topic}\nPlatform: {platform}"

    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=900,
            temperature=0.9,
        )
        return _normalize_string_list(result.get("scripts"), limit=2, min_items=2)
    except Exception:
        return _fallback_scripts(topic, platform)


def analyze_thumbnail_text(topic: str, thumbnail_text: str) -> dict[str, Any]:
    topic = _clean_text(topic)
    thumbnail_text = _clean_text(thumbnail_text)

    if not topic and not thumbnail_text:
        raise HTTPException(status_code=400, detail="Enter thumbnail text or a topic first.")

    base_text = thumbnail_text or topic
    heuristic = _score_thumbnail_text(base_text)

    instructions = (
        "You are a thumbnail text analyzer for creators. "
        "Return valid JSON with keys: text, suggestions. "
        "text should be the analyzed thumbnail text. "
        "suggestions should be an array of exactly 3 stronger thumbnail text suggestions. "
        "Each suggestion must be short, punchy, and easy to read on a thumbnail. "
        "Do not include explanations."
    )

    prompt = (
        f"Topic: {topic or 'Not provided'}\n"
        f"Thumbnail text to analyze: {base_text}"
    )

    suggestions: list[str]
    try:
        result = _call_json_model(
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=400,
            temperature=0.8,
        )
        suggestions = _normalize_string_list(result.get("suggestions"), limit=3, min_items=2)
        analyzed_text = _clean_text(result.get("text", base_text)) or base_text
    except Exception:
        analyzed_text = base_text
        suggestions = [
            f"{base_text} — Better",
            f"Why {base_text}",
            f"Stop Ignoring {base_text}",
        ]

    return {
        "text": analyzed_text,
        "score": heuristic["score"],
        "label": heuristic["label"],
        "feedback": heuristic["feedback"],
        "metrics": heuristic["metrics"],
        "suggestions": suggestions,
    }