from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.studio_usage import StudioUsageEventCreate, StudioUsageEventRead
from app.services.studio_usage_service import StudioUsageService

# Adjust this import to match your auth dependency location.
from app.auth.dependencies import get_current_user


router = APIRouter(prefix="/studio-usage", tags=["Studio Usage"])


@router.post(
    "/events",
    response_model=StudioUsageEventRead,
    status_code=status.HTTP_201_CREATED,
)
def create_studio_usage_event(
    payload: StudioUsageEventCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if not getattr(current_user, "id", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    event = StudioUsageService.log_event(
        db=db,
        user_id=current_user.id,
        payload=payload,
    )
    return event