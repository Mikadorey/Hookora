from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from app.database import Base


class WaitlistInterest(Base):
    __tablename__ = "waitlist_interests"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    current_plan = Column(String, nullable=True)
    interested_in = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    contacted = Column(Boolean, nullable=False, default=False, server_default="0")