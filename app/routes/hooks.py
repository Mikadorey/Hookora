from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.hook import Hook
from app.models.user import User
from app.services.ai_service import generate_hooks, generate_single_hook
from app.utils.plan_limits import (
    get_plan_limits,
    get_usage_snapshot,
    has_required_plan,
    reset_generation_cycle_if_needed,
)
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/hooks",
    tags=["Hooks"]
)

SUPPORTED_LANGUAGES = {"en", "es", "fr", "pt"}


def normalize_language(language: str | None) -> str:
    cleaned = (language or "en").strip().lower()
    return cleaned if cleaned in SUPPORTED_LANGUAGES else "en"


class HookGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=240)
    platform: Optional[str] = Field(default=None, max_length=50)
    template: Optional[str] = Field(default="curiosity", max_length=50)
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


class HookRegenerateRequest(BaseModel):
    template: Optional[str] = Field(default="curiosity", max_length=50)
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


class BulkHookActionRequest(BaseModel):
    hook_ids: List[int] = Field(..., min_length=1)


def serialize_hook(hook: Hook) -> dict:
    return {
        "id": hook.id,
        "topic": hook.topic,
        "content": hook.content,
        "score": hook.score,
        "platform": hook.platform,
        "is_favorite": hook.is_favorite,
        "created_at": hook.created_at.isoformat() if hook.created_at else None
    }


def build_usage_messages(usage: dict) -> dict:
    monthly_limit = usage["monthly_generation_limit"]
    monthly_count = usage["monthly_generation_count"]
    monthly_remaining = usage["monthly_generations_remaining"]
    saved_limit = usage["saved_hook_limit"]
    saved_count = usage["saved_hooks_count"]
    saved_remaining = usage["saved_hooks_remaining"]

    near_generation_limit = monthly_remaining <= 3
    reached_generation_limit = monthly_remaining <= 0

    near_saved_limit = saved_limit is not None and saved_remaining is not None and saved_remaining <= 3
    reached_saved_limit = saved_limit is not None and saved_remaining is not None and saved_remaining <= 0

    messages = []

    if reached_generation_limit:
        messages.append("You’ve reached your monthly generation limit.")
    elif near_generation_limit:
        messages.append(f"You only have {monthly_remaining} generation(s) left this month.")

    if reached_saved_limit:
        messages.append("You’ve reached your saved hook limit.")
    elif near_saved_limit:
        messages.append(f"You only have {saved_remaining} saved hook slot(s) left.")

    return {
        "near_generation_limit": near_generation_limit,
        "reached_generation_limit": reached_generation_limit,
        "near_saved_limit": near_saved_limit,
        "reached_saved_limit": reached_saved_limit,
        "messages": messages
    }


def enforce_plan_limits_for_generation(db: Session, user: User):
    reset_generation_cycle_if_needed(user)
    db.commit()
    db.refresh(user)

    limits = get_plan_limits(user.plan)
    saved_hooks_count = db.query(Hook).filter(Hook.user_id == user.id).count()
    usage = get_usage_snapshot(user, saved_hooks_count)

    if limits["saved_hook_limit"] is not None and saved_hooks_count >= limits["saved_hook_limit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": f"Free plan saved hook limit reached ({limits['saved_hook_limit']}). Upgrade to save more hooks.",
                "usage": usage,
                "upgrade_plan": "creator",
                "reason": "saved_hook_limit_reached"
            }
        )

    if user.monthly_generation_count >= limits["monthly_generation_limit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": f"Monthly generation limit reached for the {limits['plan']} plan. Upgrade to continue generating hooks.",
                "usage": usage,
                "upgrade_plan": "creator" if limits["plan"] == "free" else "pro",
                "reason": "generation_limit_reached"
            }
        )


def enforce_minimum_plan(user: User, required_plan: str, feature_name: str):
    if not has_required_plan(user.plan, required_plan):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{feature_name} requires the {required_plan.capitalize()} plan or higher."
        )


def get_user_hooks_by_ids(db: Session, user_id: int, hook_ids: List[int]) -> List[Hook]:
    hooks = (
        db.query(Hook)
        .filter(Hook.user_id == user_id, Hook.id.in_(hook_ids))
        .all()
    )

    found_ids = {hook.id for hook in hooks}
    missing_ids = [hook_id for hook_id in hook_ids if hook_id not in found_ids]

    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Some hooks were not found: {missing_ids}"
        )

    return hooks


def build_csv_content(hooks: List[Hook]) -> str:
    header = ["id", "topic", "platform", "score", "is_favorite", "created_at", "content"]

    def esc(value):
        value_str = str(value if value is not None else "")
        return f"\"{value_str.replace('\"', '\"\"')}\""

    rows = [",".join(esc(column) for column in header)]

    for hook in hooks:
        rows.append(",".join([
            esc(hook.id),
            esc(hook.topic),
            esc(hook.platform),
            esc(hook.score),
            esc(hook.is_favorite),
            esc(hook.created_at.isoformat() if hook.created_at else ""),
            esc(hook.content),
        ]))

    return "\n".join(rows)


def get_recent_hook_texts(db: Session, user_id: int, topic: str, platform: Optional[str], limit: int = 18) -> list[str]:
    query = db.query(Hook).filter(Hook.user_id == user_id, Hook.topic == topic)

    if platform:
        query = query.filter(Hook.platform == platform)

    hooks = query.order_by(Hook.created_at.desc()).limit(limit).all()
    return [hook.content for hook in hooks if hook.content]


@router.post("/generate")
def create_hook(
    payload: HookGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_plan_limits_for_generation(db, current_user)
    language = normalize_language(payload.language)

    avoid_texts = get_recent_hook_texts(
        db=db,
        user_id=current_user.id,
        topic=payload.topic,
        platform=payload.platform,
        limit=18,
    )

    hooks = generate_hooks(
        topic=payload.topic,
        platform=payload.platform,
        template=payload.template,
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
        avoid_texts=avoid_texts,
    )

    saved_hooks = []
    for hook_data in hooks:
        new_hook = Hook(
            topic=payload.topic,
            content=hook_data["text"],
            score=hook_data["score"],
            platform=hook_data["platform"],
            user_id=current_user.id,
            is_favorite=False
        )
        db.add(new_hook)
        saved_hooks.append(new_hook)

    current_user.monthly_generation_count += 1
    db.commit()

    for hook in saved_hooks:
        db.refresh(hook)

    db.refresh(current_user)
    saved_hooks_count = db.query(Hook).filter(Hook.user_id == current_user.id).count()
    usage = get_usage_snapshot(current_user, saved_hooks_count)
    usage_messages = build_usage_messages(usage)

    return {
        "topic": payload.topic,
        "template": payload.template,
        "platform": payload.platform,
        "language": language,
        "usage": usage,
        "usage_messages": usage_messages,
        "hooks": [
            {
                "id": hook.id,
                "text": hook.content,
                "score": hook.score,
                "platform": hook.platform,
                "is_favorite": hook.is_favorite,
                "created_at": hook.created_at.isoformat() if hook.created_at else None
            }
            for hook in saved_hooks
        ]
    }


@router.post("/{hook_id}/regenerate")
def regenerate_hook(
    hook_id: int,
    payload: HookRegenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_plan_limits_for_generation(db, current_user)
    language = normalize_language(payload.language)

    original_hook = (
        db.query(Hook)
        .filter(Hook.id == hook_id, Hook.user_id == current_user.id)
        .first()
    )

    if not original_hook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hook not found"
        )

    prior_hooks = get_recent_hook_texts(
        db=db,
        user_id=current_user.id,
        topic=original_hook.topic,
        platform=original_hook.platform,
        limit=20,
    )

    regenerated = generate_single_hook(
        topic=original_hook.topic,
        platform=original_hook.platform,
        template=payload.template,
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
        avoid_texts=prior_hooks,
    )

    new_hook = Hook(
        topic=original_hook.topic,
        content=regenerated["text"],
        score=regenerated["score"],
        platform=regenerated["platform"],
        user_id=current_user.id,
        is_favorite=False
    )

    db.add(new_hook)
    current_user.monthly_generation_count += 1
    db.commit()
    db.refresh(new_hook)
    db.refresh(current_user)

    saved_hooks_count = db.query(Hook).filter(Hook.user_id == current_user.id).count()
    usage = get_usage_snapshot(current_user, saved_hooks_count)
    usage_messages = build_usage_messages(usage)

    return {
        "message": "Hook regenerated successfully",
        "language": language,
        "usage": usage,
        "usage_messages": usage_messages,
        "hook": {
            "id": new_hook.id,
            "text": new_hook.content,
            "topic": new_hook.topic,
            "score": new_hook.score,
            "platform": new_hook.platform,
            "is_favorite": new_hook.is_favorite,
            "created_at": new_hook.created_at.isoformat() if new_hook.created_at else None
        }
    }


@router.patch("/{hook_id}/favorite")
def toggle_favorite_hook(
    hook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    hook = (
        db.query(Hook)
        .filter(Hook.id == hook_id, Hook.user_id == current_user.id)
        .first()
    )

    if not hook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hook not found"
        )

    hook.is_favorite = not bool(hook.is_favorite)
    db.commit()
    db.refresh(hook)

    return {
        "message": "Hook favorite updated successfully",
        "hook": serialize_hook(hook)
    }


@router.post("/bulk/favorite")
def bulk_favorite_hooks(
    payload: BulkHookActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_minimum_plan(current_user, "creator", "Bulk favorite")

    hooks = get_user_hooks_by_ids(db, current_user.id, payload.hook_ids)

    for hook in hooks:
        hook.is_favorite = True

    db.commit()

    return {
        "message": "Selected hooks favorited successfully",
        "updated_count": len(hooks)
    }


@router.post("/bulk/unfavorite")
def bulk_unfavorite_hooks(
    payload: BulkHookActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_minimum_plan(current_user, "creator", "Bulk unfavorite")

    hooks = get_user_hooks_by_ids(db, current_user.id, payload.hook_ids)

    for hook in hooks:
        hook.is_favorite = False

    db.commit()

    return {
        "message": "Selected hooks unfavorited successfully",
        "updated_count": len(hooks)
    }


@router.post("/bulk/delete")
def bulk_delete_hooks(
    payload: BulkHookActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_minimum_plan(current_user, "creator", "Bulk delete")

    hooks = get_user_hooks_by_ids(db, current_user.id, payload.hook_ids)

    deleted_count = len(hooks)
    for hook in hooks:
        db.delete(hook)

    db.commit()

    return {
        "message": "Selected hooks deleted successfully",
        "deleted_count": deleted_count
    }


@router.post("/export/csv")
def export_hooks_csv(
    payload: BulkHookActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_minimum_plan(current_user, "creator", "CSV export")

    hooks = get_user_hooks_by_ids(db, current_user.id, payload.hook_ids)
    csv_content = build_csv_content(hooks)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=hookora-hooks.csv"
        }
    )


@router.delete("/{hook_id}")
def delete_hook(
    hook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    hook = (
        db.query(Hook)
        .filter(Hook.id == hook_id, Hook.user_id == current_user.id)
        .first()
    )

    if not hook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hook not found"
        )

    db.delete(hook)
    db.commit()

    return {"message": "Hook deleted successfully", "hook_id": hook_id}


@router.get("/favorites")
def get_favorite_hooks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    hooks = (
        db.query(Hook)
        .filter(Hook.user_id == current_user.id, Hook.is_favorite.is_(True))
        .order_by(Hook.created_at.desc())
        .all()
    )

    return [serialize_hook(hook) for hook in hooks]


@router.get("/score-engine")
def hook_score_engine(
    topic: Optional[str] = None,
    platform: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Hook).filter(Hook.user_id == current_user.id)

    if topic:
        query = query.filter(Hook.topic.ilike(f"%{topic}%"))

    if platform:
        query = query.filter(Hook.platform.ilike(f"%{platform}%"))

    hooks = query.order_by(Hook.score.desc()).limit(limit).all()

    return {
        "topic": topic,
        "platform": platform,
        "hooks": [serialize_hook(h) for h in hooks]
    }


@router.get("/history")
def get_hooks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    hooks = (
        db.query(Hook)
        .filter(Hook.user_id == current_user.id)
        .order_by(Hook.created_at.desc())
        .all()
    )

    return [serialize_hook(hook) for hook in hooks]


@router.get("/analytics")
def hooks_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    avg_score_per_topic = (
        db.query(Hook.topic, func.avg(Hook.score).label("average_score"))
        .filter(Hook.user_id == current_user.id)
        .group_by(Hook.topic)
        .all()
    )

    count_per_platform = (
        db.query(Hook.platform, func.count(Hook.id).label("count"))
        .filter(Hook.user_id == current_user.id)
        .group_by(Hook.platform)
        .all()
    )

    top_hooks = (
        db.query(Hook)
        .filter(Hook.user_id == current_user.id)
        .order_by(Hook.score.desc())
        .limit(3)
        .all()
    )

    favorite_count = (
        db.query(func.count(Hook.id))
        .filter(Hook.user_id == current_user.id, Hook.is_favorite.is_(True))
        .scalar()
    )

    return {
        "average_score_per_topic": [
            {"topic": t, "average_score": float(s)}
            for t, s in avg_score_per_topic
        ],
        "count_per_platform": [
            {"platform": p, "count": c}
            for p, c in count_per_platform
        ],
        "favorite_count": favorite_count or 0,
        "top_hooks": [serialize_hook(h) for h in top_hooks]
    }


@router.get("/trends")
def hooks_trends(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    daily_avg = (
        db.query(
            func.date(Hook.created_at).label("day"),
            func.avg(Hook.score).label("avg_score")
        )
        .filter(Hook.user_id == current_user.id)
        .group_by(func.date(Hook.created_at))
        .order_by(func.date(Hook.created_at))
        .all()
    )

    platform_daily = (
        db.query(
            func.date(Hook.created_at).label("day"),
            Hook.platform,
            func.count(Hook.id).label("count")
        )
        .filter(Hook.user_id == current_user.id)
        .group_by(func.date(Hook.created_at), Hook.platform)
        .order_by(func.date(Hook.created_at))
        .all()
    )

    weekly_top_hooks = (
        db.query(Hook)
        .filter(Hook.user_id == current_user.id)
        .order_by(Hook.score.desc(), Hook.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "daily_average_score": [
            {"day": d, "avg_score": float(s)}
            for d, s in daily_avg
        ],
        "platform_daily_count": [
            {"day": d, "platform": p, "count": c}
            for d, p, c in platform_daily
        ],
        "weekly_top_hooks": [serialize_hook(h) for h in weekly_top_hooks]
    }