import random
import string
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models.referral import Referral
from app.models.user import User


REFERRAL_PREFIX = "HKR"
REFERRAL_LENGTH = 8


def generate_referral_code(db: Session) -> str:
    alphabet = string.ascii_uppercase + string.digits

    for _ in range(20):
        candidate = REFERRAL_PREFIX + "".join(random.choices(alphabet, k=REFERRAL_LENGTH))
        exists = db.query(User).filter(User.referral_code == candidate).first()
        if not exists:
            return candidate

    raise ValueError("Unable to generate a unique referral code.")


def ensure_user_has_referral_code(db: Session, user: User) -> str:
    if user.referral_code and user.referral_code.strip():
        return user.referral_code

    user.referral_code = generate_referral_code(db)
    db.commit()
    db.refresh(user)
    return user.referral_code


def get_referral_link(user: User) -> str:
    base_url = settings.frontend_app_url.rstrip("/")
    code = (user.referral_code or "").strip()
    return f"{base_url}/register?ref={code}" if code else f"{base_url}/register"


def assign_referral_to_user(db: Session, referred_user: User, code: str) -> Referral:
    normalized_code = (code or "").strip().upper()
    if not normalized_code:
        raise ValueError("Referral code is required.")

    if referred_user.referred_by_user_id or referred_user.referral_source_locked_at:
        raise ValueError("A referral has already been applied to this account.")

    referrer = db.query(User).filter(User.referral_code == normalized_code).first()
    if not referrer:
        raise ValueError("Invalid referral code.")

    if referrer.id == referred_user.id:
        raise ValueError("You cannot use your own referral code.")

    if referrer.email.lower().strip() == referred_user.email.lower().strip():
        raise ValueError("You cannot use your own referral code.")

    existing_referral = (
        db.query(Referral)
        .filter(Referral.referred_user_id == referred_user.id)
        .first()
    )
    if existing_referral:
        raise ValueError("A referral has already been applied to this account.")

    referred_user.referred_by_user_id = referrer.id
    referred_user.referral_source_locked_at = datetime.utcnow()

    referral = Referral(
        referrer_user_id=referrer.id,
        referred_user_id=referred_user.id,
        referral_code_used=normalized_code,
        status="signed_up",
        rejection_reason=None,
    )

    db.add(referral)
    db.commit()
    db.refresh(referral)
    return referral


def get_referral_summary(db: Session, user: User) -> dict[str, Any]:
    ensure_user_has_referral_code(db, user)

    referrals = db.query(Referral).filter(Referral.referrer_user_id == user.id).all()

    total_referrals = len(referrals)
    verified_referrals = len(
        [
            item
            for item in referrals
            if item.status in {
                "email_verified",
                "signup_reward_awarded",
                "paid_conversion_reward_awarded",
            }
        ]
    )
    paid_referrals = len(
        [item for item in referrals if item.paid_reward_awarded_at is not None]
    )

    total_reward_credits_earned_naira = sum(
        300 if item.signup_reward_awarded_at else 0
        for item in referrals
    ) + sum(
        700 if item.paid_reward_awarded_at else 0
        for item in referrals
    )

    return {
        "referral_code": user.referral_code,
        "referral_link": get_referral_link(user),
        "total_referrals": total_referrals,
        "verified_referrals": verified_referrals,
        "paid_referrals": paid_referrals,
        "total_reward_credits_earned_naira": total_reward_credits_earned_naira,
        "reward_credits_balance_naira": user.reward_credits_balance_naira or 0,
    }


def get_referral_history(db: Session, user: User) -> list[dict[str, Any]]:
    referrals = (
        db.query(Referral)
        .filter(Referral.referrer_user_id == user.id)
        .order_by(Referral.created_at.desc())
        .all()
    )

    items: list[dict[str, Any]] = []

    for referral in referrals:
        referred_user = db.query(User).filter(User.id == referral.referred_user_id).first()
        if referred_user:
            email = referred_user.email or ""
            local, _, domain = email.partition("@")
            masked_local = local[:2] + "***" if len(local) > 2 else "***"
            referred_user_label = f"{masked_local}@{domain}" if domain else referred_user.username
        else:
            referred_user_label = "Unknown user"

        items.append({
            "id": referral.id,
            "referred_user_label": referred_user_label,
            "status": referral.status,
            "signup_reward_awarded": referral.signup_reward_awarded_at is not None,
            "paid_reward_awarded": referral.paid_reward_awarded_at is not None,
            "created_at": referral.created_at.isoformat() if referral.created_at else None,
            "signup_reward_awarded_at": (
                referral.signup_reward_awarded_at.isoformat()
                if referral.signup_reward_awarded_at
                else None
            ),
            "paid_reward_awarded_at": (
                referral.paid_reward_awarded_at.isoformat()
                if referral.paid_reward_awarded_at
                else None
            ),
        })

    return items