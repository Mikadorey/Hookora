from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    plan = Column(String, nullable=False, default="free", server_default="free")
    monthly_generation_count = Column(Integer, nullable=False, default=0, server_default="0")
    generation_reset_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    paystack_customer_code = Column(String, nullable=True, index=True)
    paystack_subscription_code = Column(String, nullable=True, index=True)
    paystack_plan_code = Column(String, nullable=True)
    billing_status = Column(String, nullable=False, default="inactive", server_default="inactive")
    billing_current_period_end = Column(DateTime, nullable=True)

    is_email_verified = Column(Integer, nullable=False, default=0, server_default="0")
    email_verified_at = Column(DateTime, nullable=True)

    referral_code = Column(String, unique=True, index=True, nullable=True)
    referred_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    referral_source_locked_at = Column(DateTime, nullable=True)

    reward_credits_balance_naira = Column(Integer, nullable=False, default=0, server_default="0")
    wallet_balance_naira = Column(Integer, nullable=False, default=0, server_default="0")

    studio_usage_events = relationship(
        "StudioUsageEvent",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )