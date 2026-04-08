from time import perf_counter
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.studio_usage_event import StudioUsageEvent
from app.models.user import User
from app.services.ai_service import (
    analyze_thumbnail_text,
    generate_ad_copy_variants,
    generate_captions,
    generate_carousel_outlines,
    generate_ctas,
    generate_description_rewrites,
    generate_hashtags,
    generate_hook_angles,
    generate_hooks,
    generate_humanized_rewrites,
    generate_offer_positioning,
    generate_repurpose_outputs,
    generate_scripts,
    generate_titles,
    generate_viral_rewrites,
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


class RichStudioRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=240)
    platform: Optional[str] = Field(default=None, max_length=50)
    tone: Optional[str] = Field(default=None, max_length=60)
    audience: Optional[str] = Field(default=None, max_length=240)
    goal: Optional[str] = Field(default=None, max_length=240)
    offer: Optional[str] = Field(default=None, max_length=240)
    pain_point: Optional[str] = Field(default=None, max_length=240)
    desired_outcome: Optional[str] = Field(default=None, max_length=240)
    angle: Optional[str] = Field(default=None, max_length=240)
    call_to_action: Optional[str] = Field(default=None, max_length=240)
    keywords: Optional[str] = Field(default=None, max_length=600)
    brand_voice: Optional[str] = Field(default=None, max_length=240)
    content_type: Optional[str] = Field(default=None, max_length=100)
    style: Optional[str] = Field(default=None, max_length=100)
    avoid_phrases: Optional[str] = Field(default=None, max_length=600)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=5)


class DescriptionRewriteRequest(RichStudioRequest):
    original_description: str = Field(default="", max_length=5000)


class HumanizerRequest(BaseModel):
    original_text: str = Field(..., min_length=1, max_length=12000)
    tone: Optional[str] = Field(default="Natural", max_length=60)
    platform: Optional[str] = Field(default="General", max_length=50)
    audience: Optional[str] = Field(default=None, max_length=240)
    humanization_strength: Optional[str] = Field(default="Balanced", max_length=30)
    preserve_original_meaning: bool = Field(default=True)
    style_notes: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=5)


class HookAngleRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=240)
    platform: Optional[str] = Field(default=None, max_length=50)
    audience: Optional[str] = Field(default=None, max_length=240)
    goal: Optional[str] = Field(default=None, max_length=240)
    offer: Optional[str] = Field(default=None, max_length=240)
    pain_point: Optional[str] = Field(default=None, max_length=240)
    desired_outcome: Optional[str] = Field(default=None, max_length=240)
    brand_voice: Optional[str] = Field(default=None, max_length=240)
    avoid_phrases: Optional[str] = Field(default=None, max_length=600)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=5)


class ThumbnailAnalyzeRequest(BaseModel):
    topic: str = Field(default="", max_length=240)
    thumbnail_text: str = Field(default="", max_length=240)
    audience: Optional[str] = Field(default=None, max_length=240)
    angle: Optional[str] = Field(default=None, max_length=240)
    desired_outcome: Optional[str] = Field(default=None, max_length=240)
    avoid_phrases: Optional[str] = Field(default=None, max_length=600)
    extra_context: Optional[str] = Field(default=None, max_length=1500)
    language: str = Field(default="en", min_length=2, max_length=5)


class RepurposeRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=12000)
    topic: Optional[str] = Field(default=None, max_length=240)
    platform: Optional[str] = Field(default="General", max_length=50)
    audience: Optional[str] = Field(default=None, max_length=240)
    goal: Optional[str] = Field(default=None, max_length=240)
    brand_voice: Optional[str] = Field(default=None, max_length=240)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=5)


class ViralRewriteRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=12000)
    topic: Optional[str] = Field(default=None, max_length=240)
    platform: Optional[str] = Field(default="General", max_length=50)
    audience: Optional[str] = Field(default=None, max_length=240)
    goal: Optional[str] = Field(default=None, max_length=240)
    brand_voice: Optional[str] = Field(default=None, max_length=240)
    extra_context: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=5)


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


def build_metadata(payload, *, language: str, endpoint: str) -> dict:
    return {
        "topic_length": len((getattr(payload, "topic", "") or "").strip()),
        "platform": getattr(payload, "platform", None),
        "tone": getattr(payload, "tone", None),
        "audience_present": bool((getattr(payload, "audience", "") or "").strip()),
        "goal_present": bool((getattr(payload, "goal", "") or "").strip()),
        "offer_present": bool((getattr(payload, "offer", "") or "").strip()),
        "pain_point_present": bool((getattr(payload, "pain_point", "") or "").strip()),
        "desired_outcome_present": bool((getattr(payload, "desired_outcome", "") or "").strip()),
        "angle_present": bool((getattr(payload, "angle", "") or "").strip()),
        "cta_present": bool((getattr(payload, "call_to_action", "") or "").strip()),
        "keywords_present": bool((getattr(payload, "keywords", "") or "").strip()),
        "brand_voice_present": bool((getattr(payload, "brand_voice", "") or "").strip()),
        "extra_context_present": bool((getattr(payload, "extra_context", "") or "").strip()),
        "language": language,
        "endpoint": endpoint,
    }


def build_humanizer_metadata(payload: HumanizerRequest, *, language: str, endpoint: str) -> dict:
    return {
        "original_text_length": len(payload.original_text.strip()),
        "platform": payload.platform,
        "tone": payload.tone,
        "audience_present": bool((payload.audience or "").strip()),
        "style_notes_present": bool((payload.style_notes or "").strip()),
        "humanization_strength": payload.humanization_strength,
        "preserve_original_meaning": payload.preserve_original_meaning,
        "language": language,
        "endpoint": endpoint,
    }


def build_hook_angle_metadata(payload: HookAngleRequest, *, language: str, endpoint: str) -> dict:
    return {
        "topic_length": len(payload.topic.strip()),
        "platform": payload.platform,
        "audience_present": bool((payload.audience or "").strip()),
        "goal_present": bool((payload.goal or "").strip()),
        "offer_present": bool((payload.offer or "").strip()),
        "pain_point_present": bool((payload.pain_point or "").strip()),
        "desired_outcome_present": bool((payload.desired_outcome or "").strip()),
        "brand_voice_present": bool((payload.brand_voice or "").strip()),
        "avoid_phrases_present": bool((payload.avoid_phrases or "").strip()),
        "extra_context_present": bool((payload.extra_context or "").strip()),
        "language": language,
        "endpoint": endpoint,
    }


def build_source_text_metadata(
    payload,
    *,
    language: str,
    endpoint: str,
    source_field: str = "source_text",
) -> dict:
    source_text = (getattr(payload, source_field, "") or "").strip()
    return {
        "source_text_length": len(source_text),
        "topic_length": len((getattr(payload, "topic", "") or "").strip()),
        "platform": getattr(payload, "platform", None),
        "audience_present": bool((getattr(payload, "audience", "") or "").strip()),
        "goal_present": bool((getattr(payload, "goal", "") or "").strip()),
        "brand_voice_present": bool((getattr(payload, "brand_voice", "") or "").strip()),
        "extra_context_present": bool((getattr(payload, "extra_context", "") or "").strip()),
        "language": language,
        "endpoint": endpoint,
    }


@router.post("/hooks")
def studio_generate_hooks(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "viral_hook_generator"
    language = normalize_language(payload.language)

    try:
        hooks = generate_hooks(
            payload.topic,
            platform=payload.platform or "YouTube",
            template="curiosity",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "hooks": hooks
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(hooks),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/hooks"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/hooks"),
        )
        raise


@router.post("/titles")
def studio_generate_titles(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "youtube_title_optimizer"
    language = normalize_language(payload.language)

    try:
        titles = generate_titles(
            payload.topic,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            angle=payload.angle,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "language": language,
            "titles": titles
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(titles),
            generation_ms=generation_ms,
            input_mode="rich_topic",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/titles"),
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
            input_mode="rich_topic",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/titles"),
        )
        raise


@router.post("/captions")
def studio_generate_captions(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "caption_generator"
    language = normalize_language(payload.language)

    try:
        captions = generate_captions(
            payload.topic,
            payload.platform or "Instagram",
            payload.tone or "engaging",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "tone": payload.tone,
            "language": language,
            "captions": captions
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(captions),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/captions"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/captions"),
        )
        raise


@router.post("/hashtags")
def studio_generate_hashtags(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "hashtag_generator"
    language = normalize_language(payload.language)

    try:
        hashtag_sets = generate_hashtags(
            payload.topic,
            payload.platform or "Instagram",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            content_type=payload.content_type,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "sets": hashtag_sets
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(hashtag_sets),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/hashtags"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/hashtags"),
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

    try:
        rewrites = generate_description_rewrites(
            payload.topic,
            payload.platform or "YouTube",
            payload.original_description,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "rewrites": rewrites
        }

        metadata = build_metadata(payload, language=language, endpoint="/studio/descriptions")
        metadata["has_original_description"] = bool(payload.original_description.strip())
        metadata["original_description_length"] = len(payload.original_description.strip())

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(rewrites),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform_description",
            metadata_json=metadata,
        )
        return response
    except HTTPException:
        generation_ms = int((perf_counter() - started) * 1000)
        metadata = build_metadata(payload, language=language, endpoint="/studio/descriptions")
        metadata["has_original_description"] = bool(payload.original_description.strip())
        metadata["original_description_length"] = len(payload.original_description.strip())

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="failed",
            output_count=0,
            generation_ms=generation_ms,
            input_mode="rich_topic_platform_description",
            metadata_json=metadata,
        )
        raise


@router.post("/scripts")
def studio_generate_scripts(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "short_script_generator"
    language = normalize_language(payload.language)

    try:
        scripts = generate_scripts(
            payload.topic,
            payload.platform or "TikTok",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            angle=payload.angle,
            call_to_action=payload.call_to_action,
            keywords=payload.keywords,
            brand_voice=payload.brand_voice,
            style=payload.style,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "scripts": scripts
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(scripts),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/scripts"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/scripts"),
        )
        raise


@router.post("/hook-angles")
def studio_generate_hook_angles(
    payload: HookAngleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "hook_angle_generator"
    language = normalize_language(payload.language)

    try:
        angles = generate_hook_angles(
            payload.topic,
            platform=payload.platform or "General",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "angles": angles,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(angles),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_hook_angle_metadata(payload, language=language, endpoint="/studio/hook-angles"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_hook_angle_metadata(payload, language=language, endpoint="/studio/hook-angles"),
        )
        raise


@router.post("/ctas")
def studio_generate_ctas(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "cta_generator"
    language = normalize_language(payload.language)

    try:
        ctas = generate_ctas(
            payload.topic,
            payload.platform or "General",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            tone=payload.tone,
            brand_voice=payload.brand_voice,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "tone": payload.tone,
            "language": language,
            "ctas": ctas,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(ctas),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/ctas"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/ctas"),
        )
        raise


@router.post("/humanizer")
def studio_generate_humanizer_rewrites(
    payload: HumanizerRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "humanizer"
    language = normalize_language(payload.language)

    try:
        rewrites = generate_humanized_rewrites(
            payload.original_text,
            tone=payload.tone or "Natural",
            platform=payload.platform or "General",
            language=language,
            audience=payload.audience,
            humanization_strength=payload.humanization_strength or "Balanced",
            preserve_original_meaning=payload.preserve_original_meaning,
            style_notes=payload.style_notes,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "original_text": payload.original_text,
            "platform": payload.platform,
            "tone": payload.tone,
            "language": language,
            "humanization_strength": payload.humanization_strength,
            "preserve_original_meaning": payload.preserve_original_meaning,
            "rewrites": rewrites,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(rewrites),
            generation_ms=generation_ms,
            input_mode="original_text_rewrite",
            metadata_json=build_humanizer_metadata(payload, language=language, endpoint="/studio/humanizer"),
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
            input_mode="original_text_rewrite",
            metadata_json=build_humanizer_metadata(payload, language=language, endpoint="/studio/humanizer"),
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
            input_mode="topic_or_thumbnail_text",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "language": language,
                "endpoint": "/studio/thumbnail/analyze",
                "reason": "missing_topic_and_thumbnail_text",
            },
        )
        raise HTTPException(status_code=400, detail="Enter thumbnail text or a topic first.")

    try:
        result = analyze_thumbnail_text(
            payload.topic,
            payload.thumbnail_text,
            language=language,
            audience=payload.audience,
            angle=payload.angle,
            desired_outcome=payload.desired_outcome,
            avoid_phrases=payload.avoid_phrases,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "language": language,
            "analysis": {
                "text": result["text"],
                "score": result["score"],
                "label": result["label"],
                "feedback": result["feedback"],
                "metrics": result["metrics"],
            },
            "suggestions": result["suggestions"],
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(result["suggestions"]) + 1,
            generation_ms=generation_ms,
            input_mode="thumbnail_text_plus_context",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "audience_present": bool((payload.audience or "").strip()),
                "angle_present": bool((payload.angle or "").strip()),
                "desired_outcome_present": bool((payload.desired_outcome or "").strip()),
                "language": language,
                "endpoint": "/studio/thumbnail/analyze",
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
            input_mode="thumbnail_text_plus_context",
            metadata_json={
                "topic_length": len(payload.topic.strip()),
                "thumbnail_text_length": len(payload.thumbnail_text.strip()),
                "language": language,
                "endpoint": "/studio/thumbnail/analyze",
            },
        )
        raise


@router.post("/repurpose")
def studio_generate_repurpose_outputs_route(
    payload: RepurposeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "repurpose_content"
    language = normalize_language(payload.language)

    try:
        outputs = generate_repurpose_outputs(
            payload.source_text,
            topic=payload.topic,
            platform=payload.platform or "General",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "outputs": outputs,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(outputs),
            generation_ms=generation_ms,
            input_mode="source_text_repurpose",
            metadata_json=build_source_text_metadata(payload, language=language, endpoint="/studio/repurpose"),
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
            input_mode="source_text_repurpose",
            metadata_json=build_source_text_metadata(payload, language=language, endpoint="/studio/repurpose"),
        )
        raise


@router.post("/ad-copy")
def studio_generate_ad_copy_route(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "ad_copy_generator"
    language = normalize_language(payload.language)

    try:
        variants = generate_ad_copy_variants(
            payload.topic,
            platform=payload.platform or "Facebook",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "variants": variants,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(variants),
            generation_ms=generation_ms,
            input_mode="rich_topic_offer",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/ad-copy"),
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
            input_mode="rich_topic_offer",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/ad-copy"),
        )
        raise


@router.post("/carousels")
def studio_generate_carousels_route(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "carousel_generator"
    language = normalize_language(payload.language)

    try:
        carousels = generate_carousel_outlines(
            payload.topic,
            platform=payload.platform or "Instagram",
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            angle=payload.angle,
            style=payload.style,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "carousels": carousels,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(carousels),
            generation_ms=generation_ms,
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/carousels"),
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
            input_mode="rich_topic_platform",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/carousels"),
        )
        raise


@router.post("/offer-positioning")
def studio_generate_offer_positioning_route(
    payload: RichStudioRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "offer_positioning"
    language = normalize_language(payload.language)

    try:
        positioning = generate_offer_positioning(
            payload.topic,
            language=language,
            audience=payload.audience,
            goal=payload.goal,
            offer=payload.offer,
            pain_point=payload.pain_point,
            desired_outcome=payload.desired_outcome,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "language": language,
            "positioning": positioning,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(positioning),
            generation_ms=generation_ms,
            input_mode="rich_topic_offer",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/offer-positioning"),
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
            input_mode="rich_topic_offer",
            metadata_json=build_metadata(payload, language=language, endpoint="/studio/offer-positioning"),
        )
        raise


@router.post("/viral-rewrites")
def studio_generate_viral_rewrites_route(
    payload: ViralRewriteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    started = perf_counter()
    tool_name = "viral_rewriter"
    language = normalize_language(payload.language)

    try:
        rewrites = generate_viral_rewrites(
            payload.source_text,
            topic=payload.topic,
            language=language,
            platform=payload.platform or "General",
            audience=payload.audience,
            goal=payload.goal,
            brand_voice=payload.brand_voice,
            extra_context=payload.extra_context,
        )
        generation_ms = int((perf_counter() - started) * 1000)

        response = {
            "topic": payload.topic,
            "platform": payload.platform,
            "language": language,
            "rewrites": rewrites,
        }

        log_studio_usage_event(
            db=db,
            current_user=current_user,
            tool_name=tool_name,
            status="success",
            output_count=len(rewrites),
            generation_ms=generation_ms,
            input_mode="source_text_rewrite",
            metadata_json=build_source_text_metadata(payload, language=language, endpoint="/studio/viral-rewrites"),
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
            input_mode="source_text_rewrite",
            metadata_json=build_source_text_metadata(payload, language=language, endpoint="/studio/viral-rewrites"),
        )
        raise