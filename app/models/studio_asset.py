from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from app.database import Base


class StudioAsset(Base):
    __tablename__ = "studio_assets"

    id = Column(Integer, primary_key=True, index=True)
    tool_type = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    topic = Column(String, nullable=True, index=True)
    platform = Column(String, nullable=True, index=True)
    content = Column(Text, nullable=False)
    meta_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    user = relationship("User", backref="studio_assets")