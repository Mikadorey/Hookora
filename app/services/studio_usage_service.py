from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.studio_usage_event import StudioUsageEvent
from app.schemas.studio_usage import StudioUsageEventCreate, StudioUsageDashboardSummary
from app.schemas.studio_usage import ToolUsageSummaryItem, ToolUsageTimeseriesItem


class StudioUsageService:
    @staticmethod
    def log_event(
        db: Session,
        user_id: int,
        payload: StudioUsageEventCreate,
    ) -> StudioUsageEvent:
        event = StudioUsageEvent(
            user_id=user_id,
            tool_name=payload.tool_name,
            event_type=payload.event_type,
            status=payload.status,
            input_mode=payload.input_mode,
            request_id=payload.request_id,
            session_id=payload.session_id,
            output_count=payload.output_count,
            generation_ms=payload.generation_ms,
            metadata_json=payload.metadata_json,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def get_user_dashboard_summary(
        db: Session,
        user_id: int,
        days: int = 30,
    ) -> StudioUsageDashboardSummary:
        since = datetime.utcnow() - timedelta(days=days)

        base_query = db.query(StudioUsageEvent).filter(
            StudioUsageEvent.user_id == user_id,
            StudioUsageEvent.created_at >= since,
            StudioUsageEvent.event_type == "tool_run",
        )

        total_runs = base_query.count()
        successful_runs = base_query.filter(StudioUsageEvent.status == "success").count()
        failed_runs = base_query.filter(StudioUsageEvent.status == "failed").count()

        avg_generation_ms = (
            db.query(func.avg(StudioUsageEvent.generation_ms))
            .filter(
                StudioUsageEvent.user_id == user_id,
                StudioUsageEvent.created_at >= since,
                StudioUsageEvent.event_type == "tool_run",
                StudioUsageEvent.generation_ms.isnot(None),
            )
            .scalar()
        )

        total_outputs = (
            db.query(func.coalesce(func.sum(StudioUsageEvent.output_count), 0))
            .filter(
                StudioUsageEvent.user_id == user_id,
                StudioUsageEvent.created_at >= since,
                StudioUsageEvent.event_type == "tool_run",
            )
            .scalar()
        ) or 0

        tool_rows = (
            db.query(
                StudioUsageEvent.tool_name.label("tool_name"),
                func.count(StudioUsageEvent.id).label("total_runs"),
                func.sum(
                    func.case((StudioUsageEvent.status == "success", 1), else_=0)
                ).label("successful_runs"),
                func.sum(
                    func.case((StudioUsageEvent.status == "failed", 1), else_=0)
                ).label("failed_runs"),
                func.avg(StudioUsageEvent.generation_ms).label("avg_generation_ms"),
                func.coalesce(func.sum(StudioUsageEvent.output_count), 0).label("total_outputs"),
            )
            .filter(
                StudioUsageEvent.user_id == user_id,
                StudioUsageEvent.created_at >= since,
                StudioUsageEvent.event_type == "tool_run",
            )
            .group_by(StudioUsageEvent.tool_name)
            .order_by(func.count(StudioUsageEvent.id).desc(), StudioUsageEvent.tool_name.asc())
            .all()
        )

        tools = [
            ToolUsageSummaryItem(
                tool_name=row.tool_name,
                total_runs=int(row.total_runs or 0),
                successful_runs=int(row.successful_runs or 0),
                failed_runs=int(row.failed_runs or 0),
                avg_generation_ms=float(row.avg_generation_ms) if row.avg_generation_ms is not None else None,
                total_outputs=int(row.total_outputs or 0),
            )
            for row in tool_rows
        ]

        most_used_tool: Optional[str] = tools[0].tool_name if tools else None

        daily_rows = (
            db.query(
                func.date(StudioUsageEvent.created_at).label("day"),
                func.count(StudioUsageEvent.id).label("total_runs"),
            )
            .filter(
                StudioUsageEvent.user_id == user_id,
                StudioUsageEvent.created_at >= since,
                StudioUsageEvent.event_type == "tool_run",
            )
            .group_by(func.date(StudioUsageEvent.created_at))
            .order_by(func.date(StudioUsageEvent.created_at).asc())
            .all()
        )

        daily_runs = [
            ToolUsageTimeseriesItem(
                date=str(row.day),
                total_runs=int(row.total_runs or 0),
            )
            for row in daily_rows
        ]

        return StudioUsageDashboardSummary(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            avg_generation_ms=float(avg_generation_ms) if avg_generation_ms is not None else None,
            total_outputs=int(total_outputs),
            most_used_tool=most_used_tool,
            tools=tools,
            daily_runs=daily_runs,
        )