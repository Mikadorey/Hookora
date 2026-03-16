from time import perf_counter

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


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)


class TopicPlatformRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    platform: str = Field(..., min_length=2, max_length=50)


class CaptionRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    platform: str = Field(..., min_length=2, max_length=50)
    tone: str = Field(default="engaging", min_length=2, max_length=50)


class DescriptionRewriteRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    platform: str = Field(..., min_length=2, max_length=50)
    original_description: str = Field(default="", max_length=5000)


class ThumbnailAnalyzeRequest(BaseModel):
    topic: str = Field(default="", max_length=200)
    thumbnail_text: str = Field(default="", max_length=200)


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


@router.post("/titles")
def studio_generate_titles(
    payload: TopicRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "youtube_title_optimizer"

    try:
        titles = generate_titles(payload.topic)
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "titles": titles
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(titles) if isinstance(titles, list) else 1,
            generation_ms=generation_ms,
            input_mode="topic",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "endpoint": "/studio/titles",
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "endpoint": "/studio/titles",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "endpoint": "/studio/titles",
            },
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

    try:
        captions = generate_captions(payload.topic, payload.platform, payload.tone)
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "tone": payload.tone,
            "captions": captions
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(captions) if isinstance(captions, list) else 1,
            generation_ms=generation_ms,
            input_mode="topic_platform_tone",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "tone": payload.tone,
                "endpoint": "/studio/captions",
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform_tone",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "tone": payload.tone,
                "endpoint": "/studio/captions",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform_tone",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "tone": payload.tone,
                "endpoint": "/studio/captions",
            },
        )
        raise


@router.post("/hashtags")
def studio_generate_hashtags(
    payload: TopicPlatformRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "hashtag_generator"

    try:
        hashtag_sets = generate_hashtags(payload.topic, payload.platform)
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "sets": hashtag_sets
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(hashtag_sets) if isinstance(hashtag_sets, list) else 1,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/hashtags",
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/hashtags",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/hashtags",
            },
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

    try:
        rewrites = generate_description_rewrites(
            payload.topic,
            payload.platform,
            payload.original_description
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "rewrites": rewrites
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(rewrites) if isinstance(rewrites, list) else 1,
            generation_ms=generation_ms,
            input_mode="topic_platform_description",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "has_original_description": bool(payload.original_description.strip()),
                "original_description_length": len(payload.original_description.strip()),
                "endpoint": "/studio/descriptions",
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform_description",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "has_original_description": bool(payload.original_description.strip()),
                "original_description_length": len(payload.original_description.strip()),
                "endpoint": "/studio/descriptions",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform_description",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "has_original_description": bool(payload.original_description.strip()),
                "original_description_length": len(payload.original_description.strip()),
                "endpoint": "/studio/descriptions",
            },
        )
        raise


@router.post("/scripts")
def studio_generate_scripts(
    payload: TopicPlatformRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "short_script_generator"

    try:
        scripts = generate_scripts(payload.topic, payload.platform)
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "scripts": scripts
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(scripts) if isinstance(scripts, list) else 1,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/scripts",
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/scripts",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_platform",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "platform": payload.platform,
                "endpoint": "/studio/scripts",
            },
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

    if not payload.topic.strip() and not payload.thumbnail_text.strip():
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_or_thumbnail_text",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "endpoint": "/studio/thumbnail/analyze",
                "reason": "missing_topic_and_thumbnail_text",
            },
        )
        raise HTTPException(status_code=400, detail="Enter thumbnail text or a topic first.")

    try:
        result = analyze_thumbnail_text(payload.topic, payload.thumbnail_text)
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "analysis": {
                "text": result["text"],
                "score": result["score"],
                "label": result["label"],
                "feedback": result["feedback"],
                "metrics": result["metrics"]
            },
            "suggestions": result["suggestions"]
        }

        suggestion_count = len(result.get("suggestions", [])) if isinstance(result.get("suggestions"), list) else 0

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=suggestion_count,
            generation_ms=generation_ms,
            input_mode="topic_or_thumbnail_text",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "endpoint": "/studio/thumbnail/analyze",
                "score": result.get("score"),
                "label": result.get("label"),
            },
        )

        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_or_thumbnail_text",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "endpoint": "/studio/thumbnail/analyze",
            },
        )
        raise
    except Exception:
        generation_ms = int((perf_counter() - started) * 1000)
        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="topic_or_thumbnail_text",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "endpoint": "/studio/thumbnail/analyze",
            },
        )
        raise