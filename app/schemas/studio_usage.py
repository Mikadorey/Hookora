from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StudioUsageEventCreate(BaseModel):
    tool_name: str = Field(..., min_length=2, max_length=100)
    event_type: str = Field(default="tool_run", min_length=2, max_length=50)
    status: str = Field(default="success", min_length=2, max_length=20)

    input_mode: Optional[str] = Field(default=None, max_length=50)

    request_id: Optional[str] = Field(default=None, max_length=120)
    session_id: Optional[str] = Field(default=None, max_length=120)

    output_count: int = Field(default=0, ge=0)
    generation_ms: Optional[int] = Field(default=None, ge=0)

    metadata_json: Optional[Dict[str, Any]] = None


class StudioUsageEventRead(BaseModel):
    id: int
    user_id: int
    tool_name: str
    event_type: str
    status: str
    input_mode: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    output_count: int
    generation_ms: Optional[int] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ToolUsageSummaryItem(BaseModel):
    tool_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_generation_ms: Optional[float] = None
    total_outputs: int


class ToolUsageTimeseriesItem(BaseModel):
    date: str
    total_runs: int


class StudioUsageDashboardSummary(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_generation_ms: Optional[float] = None
    total_outputs: int
    most_used_tool: Optional[str] = None
    tools: List[ToolUsageSummaryItem]
    daily_runs: List[ToolUsageTimeseriesItem]