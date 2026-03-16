from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.studio_usage import StudioUsageDashboardSummary
from app.services.studio_usage_service import StudioUsageService

# Adjust this import to match your auth dependency location.
from app.auth.dependencies import get_current_user


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "/studio-analytics",
    response_model=StudioUsageDashboardSummary,
)
def get_dashboard_studio_analytics(
    days: int = Query(default=30, ge=7, le=365),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if not getattr(current_user, "id", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return StudioUsageService.get_user_dashboard_summary(
        db=db,
        user_id=current_user.id,
        days=days,
    )