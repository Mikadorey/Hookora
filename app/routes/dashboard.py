from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.hook import Hook
from app.models.studio_asset import StudioAsset
from app.models.studio_usage_event import StudioUsageEvent
from app.models.user import User
from app.utils.plan_limits import get_usage_snapshot, reset_generation_cycle_if_needed
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"]
)


def serialize_hook(hook: Hook) -> dict:
    return {
        "id": hook.id,
        "type": "hook",
        "topic": hook.topic,
        "title": hook.topic,
        "content": hook.content,
        "score": hook.score,
        "platform": hook.platform,
        "tool_type": "hooks",
        "is_favorite": hook.is_favorite,
        "created_at": hook.created_at.isoformat() if hook.created_at else None
    }


def serialize_studio_asset(asset: StudioAsset) -> dict:
    return {
        "id": asset.id,
        "type": "studio_asset",
        "topic": asset.topic,
        "title": asset.title,
        "content": asset.content,
        "score": asset.meta_json,
        "platform": asset.platform,
        "tool_type": asset.tool_type,
        "created_at": asset.created_at.isoformat() if asset.created_at else None
    }


def build_usage_messages(usage: dict) -> dict:
    monthly_remaining = usage["monthly_generations_remaining"]
    saved_limit = usage["saved_hook_limit"]
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


def build_cycle_messages(user: User, usage: dict) -> dict:
    if not user.generation_reset_date:
        return {
            "reset_date": None,
            "days_until_reset": None,
            "reset_soon": False,
            "messages": []
        }

    reset_date = user.generation_reset_date
    now = datetime.now(timezone.utc)

    if reset_date.tzinfo is None:
        reset_date = reset_date.replace(tzinfo=timezone.utc)

    delta = reset_date - now
    days_until_reset = max(0, delta.days if delta.seconds == 0 else delta.days + 1)

    messages = []

    if days_until_reset <= 3:
        messages.append(f"Your usage resets in {days_until_reset} day(s).")

    if usage["monthly_generations_remaining"] <= 0 and days_until_reset <= 3:
        messages.append("You’ve hit your limit, but your quota resets soon.")

    if usage["saved_hook_limit"] is not None and usage["saved_hooks_remaining"] == 0 and days_until_reset <= 3:
        messages.append("Your saved hook space is full, but you can upgrade now or wait for a reset cycle if your plan changes.")

    return {
        "reset_date": reset_date.isoformat(),
        "days_until_reset": days_until_reset,
        "reset_soon": days_until_reset <= 3,
        "messages": messages
    }


def build_studio_usage_analytics(
    db: Session,
    current_user: User,
    days: int = 30,
) -> dict:
    since = datetime.utcnow() - timedelta(days=days)

    usage_events = db.query(StudioUsageEvent).filter(
        StudioUsageEvent.user_id == current_user.id,
        StudioUsageEvent.event_type == "tool_run",
        StudioUsageEvent.created_at >= since,
    ).all()

    total_runs = len(usage_events)
    successful_runs = len([event for event in usage_events if event.status == "success"])
    failed_runs = len([event for event in usage_events if event.status == "failed"])

    total_outputs = sum(event.output_count or 0 for event in usage_events)

    generation_values = [
        event.generation_ms for event in usage_events
        if event.generation_ms is not None
    ]
    avg_generation_ms = (
        round(sum(generation_values) / len(generation_values), 2)
        if generation_values else 0
    )

    tool_summary_map = {}
    for event in usage_events:
        if event.tool_name not in tool_summary_map:
            tool_summary_map[event.tool_name] = {
                "tool_name": event.tool_name,
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_outputs": 0,
                "generation_values": [],
            }

        tool_summary_map[event.tool_name]["total_runs"] += 1
        tool_summary_map[event.tool_name]["total_outputs"] += event.output_count or 0

        if event.status == "success":
            tool_summary_map[event.tool_name]["successful_runs"] += 1
        elif event.status == "failed":
            tool_summary_map[event.tool_name]["failed_runs"] += 1

        if event.generation_ms is not None:
            tool_summary_map[event.tool_name]["generation_values"].append(event.generation_ms)

    tools = []
    for item in tool_summary_map.values():
        generation_list = item.pop("generation_values")
        item["avg_generation_ms"] = (
            round(sum(generation_list) / len(generation_list), 2)
            if generation_list else 0
        )
        tools.append(item)

    tools = sorted(
        tools,
        key=lambda item: (-item["total_runs"], item["tool_name"])
    )

    most_used_tool = tools[0]["tool_name"] if tools else None

    daily_rows = (
        db.query(
            func.date(StudioUsageEvent.created_at).label("day"),
            func.count(StudioUsageEvent.id).label("total_runs"),
        )
        .filter(
            StudioUsageEvent.user_id == current_user.id,
            StudioUsageEvent.event_type == "tool_run",
            StudioUsageEvent.created_at >= since,
        )
        .group_by(func.date(StudioUsageEvent.created_at))
        .order_by(func.date(StudioUsageEvent.created_at).asc())
        .all()
    )

    daily_runs = [
        {
            "date": str(row.day),
            "total_runs": int(row.total_runs or 0),
        }
        for row in daily_rows
    ]

    return {
        "range_days": days,
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": round((successful_runs / total_runs) * 100, 2) if total_runs > 0 else 0,
        "avg_generation_ms": avg_generation_ms,
        "total_outputs": total_outputs,
        "most_used_tool": most_used_tool,
        "tools": tools,
        "daily_runs": daily_runs,
    }


def get_admin_email_set() -> set[str]:
    raw = settings.admin_emails or ""
    return {email.strip().lower() for email in raw.split(",") if email.strip()}


def is_admin_email(email: str) -> bool:
    return email.lower().strip() in get_admin_email_set()


def _normalize_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value


def build_founder_overview(db: Session) -> dict:
    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    users = (
        db.query(User)
        .order_by(User.created_at.desc().nullslast(), User.id.desc())
        .all()
    )

    total_users = len(users)
    new_users_7d = 0
    new_users_30d = 0
    free_users = 0
    creator_users = 0
    pro_users = 0
    paid_users = 0

    for user in users:
        created_at = _normalize_datetime(user.created_at)
        if created_at:
            if created_at >= seven_days_ago:
                new_users_7d += 1
            if created_at >= thirty_days_ago:
                new_users_30d += 1

        normalized_plan = (user.plan or "free").lower()
        if normalized_plan == "creator":
            creator_users += 1
            paid_users += 1
        elif normalized_plan == "pro":
            pro_users += 1
            paid_users += 1
        else:
            free_users += 1

    active_user_ids = {
        user_id
        for (user_id,) in db.query(Hook.user_id).distinct().all()
        if user_id is not None
    }
    active_users = len(active_user_ids)
    conversion_rate = round((paid_users / total_users) * 100, 2) if total_users > 0 else 0.0

    recent_signups = []
    for user in users[:8]:
        recent_signups.append({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "plan": user.plan,
            "billing_status": getattr(user, "billing_status", "inactive"),
            "created_at": user.created_at.isoformat() if user.created_at else None
        })

    focus = []
    if new_users_7d == 0:
        focus.append("Push user acquisition this week.")
    if paid_users == 0:
        focus.append("Focus on first paid conversions.")
    if active_users < total_users and total_users > 0:
        focus.append("Improve activation and retention for new users.")
    if total_users > 0 and conversion_rate < 10:
        focus.append("Test stronger upgrade prompts and pricing copy.")
    if not focus:
        focus.append("Maintain momentum and double down on what is working.")

    return {
        "total_users": total_users,
        "new_users_7d": new_users_7d,
        "new_users_30d": new_users_30d,
        "active_users": active_users,
        "paid_users": paid_users,
        "free_users": free_users,
        "creator_users": creator_users,
        "pro_users": pro_users,
        "signup_to_paid_conversion_rate": conversion_rate,
        "recent_signups": recent_signups,
        "weekly_focus": focus,
    }


@router.get("/")
def get_dashboard_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    reset_generation_cycle_if_needed(current_user)
    db.commit()
    db.refresh(current_user)

    hooks = db.query(Hook).filter(Hook.user_id == current_user.id).all()
    studio_assets = db.query(StudioAsset).filter(StudioAsset.user_id == current_user.id).all()

    platform_counts = {}
    for hook in hooks:
        platform_counts[hook.platform] = platform_counts.get(hook.platform, 0) + 1

    studio_tool_counts = {}
    for asset in studio_assets:
        studio_tool_counts[asset.tool_type] = studio_tool_counts.get(asset.tool_type, 0) + 1

    top_hooks = sorted(hooks, key=lambda h: h.score or 0, reverse=True)[:5]
    recent_hooks = sorted(
        hooks,
        key=lambda h: h.created_at or datetime.min,
        reverse=True
    )[:5]

    recent_studio_assets = sorted(
        studio_assets,
        key=lambda a: a.created_at or datetime.min,
        reverse=True
    )[:8]

    favorite_hooks = sorted(
        [hook for hook in hooks if hook.is_favorite],
        key=lambda h: h.created_at or datetime.min,
        reverse=True
    )[:10]

    avg_score = sum(h.score or 0 for h in hooks) / len(hooks) if hooks else 0
    usage = get_usage_snapshot(current_user, len(hooks))
    usage_messages = build_usage_messages(usage)
    cycle_messages = build_cycle_messages(current_user, usage)
    studio_usage_analytics = build_studio_usage_analytics(db, current_user, days=30)

    combined_activity = [serialize_hook(hook) for hook in hooks] + [serialize_studio_asset(asset) for asset in studio_assets]
    combined_activity = sorted(
        combined_activity,
        key=lambda item: item["created_at"] or "",
        reverse=True
    )[:12]

    top_tool_types = sorted(
        [{"tool_type": key, "count": value} for key, value in studio_tool_counts.items()],
        key=lambda item: item["count"],
        reverse=True
    )

    founder_overview = build_founder_overview(db) if is_admin_email(current_user.email) else None

    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "plan": current_user.plan,
            "is_admin": is_admin_email(current_user.email),
        },
        "usage": usage,
        "usage_messages": usage_messages,
        "cycle_messages": cycle_messages,
        "metrics": {
            "total_hooks": len(hooks),
            "total_studio_assets": len(studio_assets),
            "total_saved_items": len(hooks) + len(studio_assets),
            "average_score": avg_score,
            "favorite_hooks": len(favorite_hooks),
            "platform_counts": platform_counts,
            "studio_tool_counts": studio_tool_counts,
            "total_studio_runs": studio_usage_analytics["total_runs"],
            "successful_studio_runs": studio_usage_analytics["successful_runs"],
            "failed_studio_runs": studio_usage_analytics["failed_runs"],
            "most_used_studio_tool": studio_usage_analytics["most_used_tool"],
        },
        "studio_usage_analytics": studio_usage_analytics,
        "top_hooks": [serialize_hook(hook) for hook in top_hooks],
        "recent_hooks": [serialize_hook(hook) for hook in recent_hooks],
        "favorite_hooks": [serialize_hook(hook) for hook in favorite_hooks],
        "recent_studio_assets": [serialize_studio_asset(asset) for asset in recent_studio_assets],
        "combined_activity": combined_activity,
        "top_tool_types": top_tool_types,
        "founder_overview": founder_overview,
    }


@router.get("/studio-analytics")
def get_studio_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return build_studio_usage_analytics(db, current_user, days=30)