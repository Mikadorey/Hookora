from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.database import Base


class Hook(Base):
    __tablename__ = "hooks"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    content = Column(String, nullable=False)
    score = Column(Float, default=0.0)
    platform = Column(String, nullable=False)
    is_favorite = Column(Boolean, nullable=False, default=False, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", backref="hooks")