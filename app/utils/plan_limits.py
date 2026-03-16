from datetime import datetime, timedelta, timezone

FREE_PLAN_GENERATION_LIMIT = 20
FREE_PLAN_SAVED_HOOK_LIMIT = 10

CREATOR_PLAN_GENERATION_LIMIT = 300
PRO_PLAN_GENERATION_LIMIT = 1500

PLAN_RANK = {
    "free": 0,
    "creator": 1,
    "pro": 2
}

STRIPE_PRICE_TO_PLAN = {
    "creator": "creator",
    "pro": "pro"
}


def get_plan_limits(plan: str) -> dict:
    normalized_plan = (plan or "free").lower()

    if normalized_plan == "creator":
        return {
            "plan": "creator",
            "monthly_generation_limit": CREATOR_PLAN_GENERATION_LIMIT,
            "saved_hook_limit": None
        }

    if normalized_plan == "pro":
        return {
            "plan": "pro",
            "monthly_generation_limit": PRO_PLAN_GENERATION_LIMIT,
            "saved_hook_limit": None
        }

    return {
        "plan": "free",
        "monthly_generation_limit": FREE_PLAN_GENERATION_LIMIT,
        "saved_hook_limit": FREE_PLAN_SAVED_HOOK_LIMIT
    }


def has_required_plan(user_plan: str, required_plan: str) -> bool:
    current_rank = PLAN_RANK.get((user_plan or "free").lower(), 0)
    required_rank = PLAN_RANK.get((required_plan or "free").lower(), 0)
    return current_rank >= required_rank


def get_next_reset_date() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=30)


def reset_generation_cycle_if_needed(user) -> bool:
    now = datetime.now(timezone.utc)

    if user.generation_reset_date is None:
        user.monthly_generation_count = 0
        user.generation_reset_date = get_next_reset_date()
        return True

    reset_date = user.generation_reset_date

    if reset_date.tzinfo is None:
        reset_date = reset_date.replace(tzinfo=timezone.utc)

    if now >= reset_date:
        user.monthly_generation_count = 0
        user.generation_reset_date = get_next_reset_date()
        return True

    return False


def get_usage_snapshot(user, saved_hooks_count: int) -> dict:
    limits = get_plan_limits(user.plan)

    monthly_limit = limits["monthly_generation_limit"]
    saved_limit = limits["saved_hook_limit"]

    return {
        "plan": limits["plan"],
        "monthly_generation_count": user.monthly_generation_count,
        "monthly_generation_limit": monthly_limit,
        "monthly_generations_remaining": max(0, monthly_limit - user.monthly_generation_count),
        "saved_hooks_count": saved_hooks_count,
        "saved_hook_limit": saved_limit,
        "saved_hooks_remaining": None if saved_limit is None else max(0, saved_limit - saved_hooks_count)
    }