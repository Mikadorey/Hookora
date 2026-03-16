from sqlalchemy import Column, Integer, String
from app.database import Base

class HookHistory(Base):

    __tablename__ = "hook_history"

    id = Column(Integer, primary_key=True, index=True)

    hook = Column(String, nullable=False)

    platform = Column(String, nullable=False)