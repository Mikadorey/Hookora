from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.hook import Hook
from app.models.user import User
from app.models.waitlist_interest import WaitlistInterest
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/waitlist",
    tags=["Waitlist"]
)


class WaitlistInterestRequest(BaseModel):
    email: EmailStr
    interested_in: str = Field(default="creator")
    notes: str | None = Field(default=None, max_length=500)


def get_admin_email_set() -> set[str]:
    raw = settings.admin_emails or ""
    return {email.strip().lower() for email in raw.split(",") if email.strip()}


def enforce_admin_access(current_user: User):
    admin_emails = get_admin_email_set()
    if current_user.email.lower().strip() not in admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )


def _normalize_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value


@router.post("/interest")
def capture_waitlist_interest(
    payload: WaitlistInterestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    normalized_email = payload.email.lower().strip()

    existing = (
        db.query(WaitlistInterest)
        .filter(WaitlistInterest.email == normalized_email)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already on the billing waitlist."
        )

    entry = WaitlistInterest(
        email=normalized_email,
        username=current_user.username,
        current_plan=current_user.plan,
        interested_in=payload.interested_in.strip().lower(),
        notes=(payload.notes or "").strip() or None
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "message": "You’ve been added to the billing interest waitlist.",
        "entry": {
            "id": entry.id,
            "email": entry.email,
            "username": entry.username,
            "current_plan": entry.current_plan,
            "interested_in": entry.interested_in,
            "notes": entry.notes,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "contacted": entry.contacted
        }
    }


@router.get("/admin/leads")
def get_waitlist_leads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_admin_access(current_user)

    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    leads = (
        db.query(WaitlistInterest)
        .order_by(WaitlistInterest.created_at.desc())
        .all()
    )

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
    for user in users[:10]:
        recent_signups.append({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "plan": user.plan,
            "billing_status": getattr(user, "billing_status", "inactive"),
            "created_at": user.created_at.isoformat() if user.created_at else None
        })

    return {
        "count": len(leads),
        "leads": [
            {
                "id": lead.id,
                "email": lead.email,
                "username": lead.username,
                "current_plan": lead.current_plan,
                "interested_in": lead.interested_in,
                "notes": lead.notes,
                "created_at": lead.created_at.isoformat() if lead.created_at else None,
                "contacted": lead.contacted
            }
            for lead in leads
        ],
        "user_analytics": {
            "total_users": total_users,
            "new_users_7d": new_users_7d,
            "new_users_30d": new_users_30d,
            "active_users": active_users,
            "paid_users": paid_users,
            "free_users": free_users,
            "creator_users": creator_users,
            "pro_users": pro_users,
            "signup_to_paid_conversion_rate": conversion_rate,
            "recent_signups": recent_signups
        }
    }