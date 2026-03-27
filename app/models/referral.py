from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from app.database import Base


class Referral(Base):
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, index=True)

    referrer_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    referred_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)

    referral_code_used = Column(String, nullable=False, index=True)

    status = Column(String, nullable=False, default="signed_up", server_default="signed_up")
    rejection_reason = Column(String, nullable=True)

    signup_reward_awarded_at = Column(DateTime, nullable=True)
    paid_reward_awarded_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("referred_user_id", name="uq_referrals_referred_user_id"),
    )