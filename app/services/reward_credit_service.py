from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.referral import Referral
from app.models.reward_credit_ledger import RewardCreditLedger
from app.models.user import User

SIGNUP_VERIFIED_REWARD_NAIRA = 300
PAID_CONVERSION_REWARD_NAIRA = 700


def compute_reward_credit_balance(db: Session, user_id: int) -> int:
    total = (
        db.query(func.coalesce(func.sum(RewardCreditLedger.amount_naira), 0))
        .filter(
            RewardCreditLedger.user_id == user_id,
            RewardCreditLedger.status == "posted",
        )
        .scalar()
    )
    return int(total or 0)


def refresh_reward_credit_balance(db: Session, user_id: int) -> int:
    balance = compute_reward_credit_balance(db, user_id)
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.reward_credits_balance_naira = balance
        db.commit()
        db.refresh(user)
    return balance


def post_reward_credit_entry(
    db: Session,
    user_id: int,
    entry_type: str,
    amount_naira: int,
    reference: str | None = None,
    metadata: dict | None = None,
) -> RewardCreditLedger:
    if reference:
        existing = (
            db.query(RewardCreditLedger)
            .filter(
                RewardCreditLedger.user_id == user_id,
                RewardCreditLedger.entry_type == entry_type,
                RewardCreditLedger.reference == reference,
            )
            .first()
        )
        if existing:
            return existing

    entry = RewardCreditLedger(
        user_id=user_id,
        entry_type=entry_type,
        amount_naira=amount_naira,
        status="posted",
        reference=reference,
        metadata_json=metadata or {},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    refresh_reward_credit_balance(db, user_id)
    return entry


def award_signup_verified_reward(db: Session, referred_user_id: int) -> bool:
    referred_user = db.query(User).filter(User.id == referred_user_id).first()
    if not referred_user or not int(referred_user.is_email_verified or 0):
        return False

    referral = (
        db.query(Referral)
        .filter(Referral.referred_user_id == referred_user_id)
        .first()
    )
    if not referral:
        return False

    if referral.signup_reward_awarded_at is not None:
        return False

    referrer = db.query(User).filter(User.id == referral.referrer_user_id).first()
    if not referrer:
        return False

    reference = f"referral-signup-verified:{referral.id}"
    post_reward_credit_entry(
        db=db,
        user_id=referrer.id,
        entry_type="referral_signup_verified_bonus",
        amount_naira=SIGNUP_VERIFIED_REWARD_NAIRA,
        reference=reference,
        metadata={
            "referral_id": referral.id,
            "referred_user_id": referred_user_id,
        },
    )

    referral.status = "signup_reward_awarded"
    referral.signup_reward_awarded_at = datetime.utcnow()
    db.commit()
    db.refresh(referral)
    return True


def award_paid_conversion_reward(
    db: Session,
    referred_user_id: int,
    payment_reference: str,
) -> bool:
    referral = (
        db.query(Referral)
        .filter(Referral.referred_user_id == referred_user_id)
        .first()
    )
    if not referral:
        return False

    if referral.paid_reward_awarded_at is not None:
        return False

    referrer = db.query(User).filter(User.id == referral.referrer_user_id).first()
    if not referrer:
        return False

    reference = f"referral-paid-conversion:{referral.id}:{payment_reference}"
    post_reward_credit_entry(
        db=db,
        user_id=referrer.id,
        entry_type="referral_paid_conversion_bonus",
        amount_naira=PAID_CONVERSION_REWARD_NAIRA,
        reference=reference,
        metadata={
            "referral_id": referral.id,
            "referred_user_id": referred_user_id,
            "payment_reference": payment_reference,
        },
    )

    referral.status = "paid_conversion_reward_awarded"
    referral.paid_reward_awarded_at = datetime.utcnow()
    db.commit()
    db.refresh(referral)
    return True


def get_available_reward_credits(db: Session, user_id: int) -> int:
    return max(0, compute_reward_credit_balance(db, user_id))


def preview_reward_credits_application(
    db: Session,
    user_id: int,
    plan_amount_naira: int,
) -> dict:
    available = get_available_reward_credits(db, user_id)
    applied = min(available, max(0, plan_amount_naira))
    remaining = max(0, plan_amount_naira - applied)

    return {
        "reward_credits_available_naira": available,
        "reward_credits_applied_naira": applied,
        "card_due_naira": remaining,
    }


def consume_reward_credits(
    db: Session,
    user_id: int,
    amount_naira: int,
    reference: str,
    metadata: dict | None = None,
) -> bool:
    amount = int(amount_naira or 0)
    if amount <= 0:
        return False

    available = get_available_reward_credits(db, user_id)
    if amount > available:
        amount = available

    if amount <= 0:
        return False

    post_reward_credit_entry(
        db=db,
        user_id=user_id,
        entry_type="billing_applied",
        amount_naira=-amount,
        reference=reference,
        metadata=metadata or {},
    )
    return True