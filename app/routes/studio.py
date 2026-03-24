from time import perf_counter
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.studio_usage_event import StudioUsageEvent
from app.models.user import User
from app.services.ai_service import (
    analyze_thumbnail_text,
    generate_captions,
    generate_description_rewrites,
    generate_hashtags,
    generate_scripts,
    generate_titles,
)
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/studio",
    tags=["Studio"]
)

SUPPORTED_LANGUAGES = {"en", "es", "fr", "pt"}


def normalize_language(language: str | None) -> str:
    cleaned = (language or "en").strip().lower()
    return cleaned if cleaned in SUPPORTED_LANGUAGES else "en"


class StudioBaseRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=240)
    platform: Optional[str] = Field(default=None, max_length=50)
    language: str = Field(default="en", min_length=2, max_length=5)

    audience: Optional[str] = Field(default=None, max_length=200)
    goal: Optional[str] = Field(default=None, max_length=200)
    product_or_offer: Optional[str] = Field(default=None, max_length=200)
    pain_point: Optional[str] = Field(default=None, max_length=240)
    desired_outcome: Optional[str] = Field(default=None, max_length=240)
    angle: Optional[str] = Field(default=None, max_length=200)
    call_to_action: Optional[str] = Field(default=None, max_length=200)
    keywords: Optional[list[str] | str] = None
    brand_voice: Optional[str] = Field(default=None, max_length=200)
    content_type: Optional[str] = Field(default=None, max_length=100)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    examples_to_avoid: Optional[list[str] | str] = None
    banned_phrases: Optional[list[str] | str] = None


class TitleRequest(StudioBaseRequest):
    pass


class CaptionRequest(StudioBaseRequest):
    platform: str = Field(..., min_length=2, max_length=50)
    tone: str = Field(default="engaging", min_length=2, max_length=50)


class HashtagRequest(StudioBaseRequest):
    platform: str = Field(..., min_length=2, max_length=50)


class DescriptionRewriteRequest(StudioBaseRequest):
    platform: str = Field(..., min_length=2, max_length=50)
    original_description: str = Field(default="", max_length=5000)


class ScriptRequest(StudioBaseRequest):
    platform: str = Field(..., min_length=2, max_length=50)


class ThumbnailAnalyzeRequest(BaseModel):
    topic: str = Field(default="", max_length=240)
    thumbnail_text: str = Field(default="", max_length=200)
    language: str = Field(default="en", min_length=2, max_length=5)

    audience: Optional[str] = Field(default=None, max_length=200)
    goal: Optional[str] = Field(default=None, max_length=200)
    pain_point: Optional[str] = Field(default=None, max_length=240)
    desired_outcome: Optional[str] = Field(default=None, max_length=240)
    angle: Optional[str] = Field(default=None, max_length=200)
    keywords: Optional[list[str] | str] = None
    brand_voice: Optional[str] = Field(default=None, max_length=200)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    banned_phrases: Optional[list[str] | str] = None


def log_studio_usage_event(
    db: Session,
    current_user: User,
    tool_name: str,
    status: str,
    output_count: int = 0,
    generation_ms: int | None = None,
    input_mode: str | None = None,
    metadata_json: dict | None = None,
) -> None:
    event = StudioUsageEvent(
        user_id=current_user.id,
        tool_name=tool_name,
        event_type="tool_run",
        status=status,
        input_mode=input_mode,
        output_count=output_count,
        generation_ms=generation_ms,
        metadata_json=metadata_json or {},
    )
    db.add(event)
    db.commit()


def build_context_metadata(payload: BaseModel, endpoint: str) -> dict[str, Any]:
    data = payload.model_dump()
    return {
        "endpoint": endpoint,
        "language": data.get("language"),
        "platform": data.get("platform"),
        "topic_length": len((data.get("topic") or "").strip()),
        "has_audience": bool((data.get("audience") or "").strip()) if isinstance(data.get("audience"), str) else bool(data.get("audience")),
        "has_goal": bool((data.get("goal") or "").strip()) if isinstance(data.get("goal"), str) else bool(data.get("goal")),
        "has_product_or_offer": bool((data.get("product_or_offer") or "").strip()) if isinstance(data.get("product_or_offer"), str) else bool(data.get("product_or_offer")),
        "has_pain_point": bool((data.get("pain_point") or "").strip()) if isinstance(data.get("pain_point"), str) else bool(data.get("pain_point")),
        "has_desired_outcome": bool((data.get("desired_outcome") or "").strip()) if isinstance(data.get("desired_outcome"), str) else bool(data.get("desired_outcome")),
        "has_angle": bool((data.get("angle") or "").strip()) if isinstance(data.get("angle"), str) else bool(data.get("angle")),
        "has_cta": bool((data.get("call_to_action") or "").strip()) if isinstance(data.get("call_to_action"), str) else bool(data.get("call_to_action")),
        "has_keywords": bool(data.get("keywords")),
        "has_brand_voice": bool((data.get("brand_voice") or "").strip()) if isinstance(data.get("brand_voice"), str) else bool(data.get("brand_voice")),
        "has_content_type": bool((data.get("content_type") or "").strip()) if isinstance(data.get("content_type"), str) else bool(data.get("content_type")),
        "has_extra_context": bool((data.get("extra_context") or "").strip()) if isinstance(data.get("extra_context"), str) else bool(data.get("extra_context")),
        "has_examples_to_avoid": bool(data.get("examples_to_avoid")),
        "has_banned_phrases": bool(data.get("banned_phrases")),
    }


@router.post("/titles")
def studio_generate_titles(
    payload: TitleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "youtube_title_optimizer"
    language = normalize_language(payload.language)
    metadata = build_context_metadata(payload, "/studio/titles")

    try:
        titles = generate_titles(
            topic=payload.topic,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            product_or_offer=payload.product_or_offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            content_type=payload.content_type,
            extra_context=payload.extra_context,
            examples_to_avoid=payload.examples_to_avoid,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(titles),
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )

        return {
            "topic": payload.topic,
            "language": language,
            "titles": titles
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )
        raise


@router.post("/captions")
def studio_generate_captions(
    payload: CaptionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "caption_generator"
    language = normalize_language(payload.language)
    metadata = build_context_metadata(payload, "/studio/captions")
    metadata["tone"] = payload.tone

    try:
        captions = generate_captions(
            topic=payload.topic,
            platform=payload.platform,
            tone=payload.tone,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            product_or_offer=payload.product_or_offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            content_type=payload.content_type,
            extra_context=payload.extra_context,
            examples_to_avoid=payload.examples_to_avoid,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(captions),
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )

        return {
            "topic": payload.topic,
            "platform": payload.platform,
            "tone": payload.tone,
            "language": language,
            "captions": captions
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )
        raise


@router.post("/hashtags")
def studio_generate_hashtags(
    payload: HashtagRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "hashtag_generator"
    language = normalize_language(payload.language)
    metadata = build_context_metadata(payload, "/studio/hashtags")

    try:
        hashtag_sets = generate_hashtags(
            topic=payload.topic,
            platform=payload.platform,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            product_or_offer=payload.product_or_offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            content_type=payload.content_type,
            extra_context=payload.extra_context,
            examples_to_avoid=payload.examples_to_avoid,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(hashtag_sets),
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )

        return {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "sets": hashtag_sets
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )
        raise


@router.post("/descriptions")
def studio_generate_descriptions(
    payload: DescriptionRewriteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "description_rewriter"
    language = normalize_language(payload.language)
    metadata = build_context_metadata(payload, "/studio/descriptions")
    metadata["has_original_description"] = bool(payload.original_description.strip())
    metadata["original_description_length"] = len(payload.original_description.strip())

    try:
        rewrites = generate_description_rewrites(
            topic=payload.topic,
            platform=payload.platform,
            original_description=payload.original_description,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            product_or_offer=payload.product_or_offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            content_type=payload.content_type,
            extra_context=payload.extra_context,
            examples_to_avoid=payload.examples_to_avoid,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(rewrites),
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )

        return {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "rewrites": rewrites
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )
        raise


@router.post("/scripts")
def studio_generate_scripts(
    payload: ScriptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "short_script_generator"
    language = normalize_language(payload.language)
    metadata = build_context_metadata(payload, "/studio/scripts")

    try:
        scripts = generate_scripts(
            topic=payload.topic,
            platform=payload.platform,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            product_or_offer=payload.product_or_offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            content_type=payload.content_type,
            extra_context=payload.extra_context,
            examples_to_avoid=payload.examples_to_avoid,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(scripts),
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )

        return {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "scripts": scripts
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_context",
            metadata_json=metadata,
        )
        raise


@router.post("/thumbnail/analyze")
def studio_analyze_thumbnail(
    payload: ThumbnailAnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "thumbnail_text_analyzer"
    language = normalize_language(payload.language)

    if not payload.topic.strip() and not payload.thumbnail_text.strip():
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="thumbnail_analysis",
            metadata_json={
                "endpoint": "/studio/thumbnail/analyze",
                "topic_length": 0,
                "thumbnail_text_length": 0,
                "reason": "missing_topic_and_thumbnail_text",
            },
        )
        raise HTTPException(status_code=400, detail="Enter thumbnail text or a topic first.")

    metadata = {
        "endpoint": "/studio/thumbnail/analyze",
        "language": language,
        "topic_length": len(payload.topic.strip()),
        "thumbnail_text_length": len(payload.thumbnail_text.strip()),
        "has_audience": bool((payload.audience or "").strip()),
        "has_goal": bool((payload.goal or "").strip()),
        "has_pain_point": bool((payload.pain_point or "").strip()),
        "has_desired_outcome": bool((payload.desired_outcome or "").strip()),
        "has_angle": bool((payload.angle or "").strip()),
        "has_keywords": bool(payload.keywords),
        "has_brand_voice": bool((payload.brand_voice or "").strip()),
        "has_extra_context": bool((payload.extra_context or "").strip()),
        "has_banned_phrases": bool(payload.banned_phrases),
    }

    try:
        result = analyze_thumbnail_text(
            topic=payload.topic,
            thumbnail_text=payload.thumbnail_text,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
            banned_phrases=payload.banned_phrases,
        )
        generation_ms = int((perf_counter() - started) * 1000)
        suggestion_count = len(result.get("suggestions", [])) if isinstance(result.get("suggestions"), list) else 0

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=suggestion_count,
            generation_ms=generation_ms,
            input_mode="thumbnail_analysis",
            metadata_json={**metadata, "score": result.get("score"), "label": result.get("label")},
        )

        return {
            "topic": payload.topic,
            "language": language,
            "analysis": {
                "text": result["text"],
                "score": result["score"],
                "label": result["label"],
                "feedback": result["feedback"],
                "metrics": result["metrics"]
            },
            "suggestions": result["suggestions"]
        }
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="thumbnail_analysis",
            metadata_json=metadata,
        )
        raise