from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.reward_credit_ledger import RewardCreditLedger
from app.models.user import User
from app.schemas.referral import ApplyReferralCodeRequest
from app.services.referral_service import (
    assign_referral_to_user,
    ensure_user_has_referral_code,
    get_referral_history,
    get_referral_summary,
)
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/referrals",
    tags=["Referrals"],
)


@router.get("/me")
def get_my_referral_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensure_user_has_referral_code(db, current_user)
    db.refresh(current_user)
    return get_referral_summary(db, current_user)


@router.get("/history")
def get_my_referral_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensure_user_has_referral_code(db, current_user)
    return get_referral_history(db, current_user)


@router.post("/apply-code")
def apply_referral_code(
    payload: ApplyReferralCodeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        referral = assign_referral_to_user(db, current_user, payload.code)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    db.refresh(current_user)
    return {
        "message": "Referral code applied successfully.",
        "referral_id": referral.id,
        "referred_by_user_id": current_user.referred_by_user_id,
    }


@router.get("/reward-ledger")
def get_reward_ledger(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entries = (
        db.query(RewardCreditLedger)
        .filter(RewardCreditLedger.user_id == current_user.id)
        .order_by(RewardCreditLedger.created_at.desc())
        .all()
    )

    return [
        {
            "id": entry.id,
            "entry_type": entry.entry_type,
            "amount_naira": entry.amount_naira,
            "status": entry.status,
            "reference": entry.reference,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
        }
        for entry in entries
    ]