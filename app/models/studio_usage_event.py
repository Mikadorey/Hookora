from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship

from app.database import Base


class StudioUsageEvent(Base):
    __tablename__ = "studio_usage_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    tool_name = Column(String(100), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, default="tool_run", index=True)

    status = Column(String(20), nullable=False, default="success", index=True)
    input_mode = Column(String(50), nullable=True)

    request_id = Column(String(120), nullable=True, index=True)
    session_id = Column(String(120), nullable=True, index=True)

    output_count = Column(Integer, nullable=False, default=0)
    generation_ms = Column(Integer, nullable=True)

    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="studio_usage_events")

    __table_args__ = (
        Index("ix_studio_usage_user_tool_created", "user_id", "tool_name", "created_at"),
        Index("ix_studio_usage_tool_status_created", "tool_name", "status", "created_at"),
    )