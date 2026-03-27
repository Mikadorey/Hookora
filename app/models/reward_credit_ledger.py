from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, func
from app.database import Base


class RewardCreditLedger(Base):
    __tablename__ = "reward_credit_ledger"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    entry_type = Column(String, nullable=False, index=True)
    amount_naira = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="posted", server_default="posted", index=True)

    reference = Column(String, nullable=True, index=True)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())