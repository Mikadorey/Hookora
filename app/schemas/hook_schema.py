from pydantic import BaseModel
from datetime import datetime

# Schema for creating a hook history entry
class HookHistoryCreate(BaseModel):
    prompt: str
    generated_text: str

# Schema for reading a hook history entry
class HookHistoryRead(BaseModel):
    id: int
    prompt: str
    generated_text: str
    created_at: datetime

    model_config = {
        "from_attributes": True  # replaces orm_mode in Pydantic V2
    }