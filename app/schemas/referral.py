from typing import Optional

from pydantic import BaseModel, Field


class ApplyReferralCodeRequest(BaseModel):
    code: str = Field(..., min_length=3, max_length=64)


class ReferralSummaryResponse(BaseModel):
    referral_code: str
    referral_link: str
    total_referrals: int
    verified_referrals: int
    paid_referrals: int
    total_reward_credits_earned_naira: int
    reward_credits_balance_naira: int


class ReferralHistoryItem(BaseModel):
    id: int
    referred_user_label: str
    status: str
    signup_reward_awarded: bool
    paid_reward_awarded: bool
    created_at: Optional[str] = None
    signup_reward_awarded_at: Optional[str] = None
    paid_reward_awarded_at: Optional[str] = None


class RewardCreditLedgerItem(BaseModel):
    id: int
    entry_type: str
    amount_naira: int
    status: str
    reference: Optional[str] = None
    created_at: Optional[str] = None