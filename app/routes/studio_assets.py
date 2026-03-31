import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.studio_asset import StudioAsset
from app.models.user import User
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/studio-assets",
    tags=["Studio Assets"]
)


class StudioAssetCreateRequest(BaseModel):
    tool_type: str = Field(..., min_length=2, max_length=50)
    title: str = Field(..., min_length=2, max_length=200)
    topic: Optional[str] = Field(default=None, max_length=200)
    platform: Optional[str] = Field(default=None, max_length=50)
    content: str = Field(..., min_length=1)
    meta: Optional[dict] = None


def serialize_asset(asset: StudioAsset) -> dict:
    parsed_meta = None

    if asset.meta_json:
        try:
            parsed_meta = json.loads(asset.meta_json)
        except json.JSONDecodeError:
            parsed_meta = None

    return {
        "id": asset.id,
        "tool_type": asset.tool_type,
        "title": asset.title,
        "topic": asset.topic,
        "platform": asset.platform,
        "content": asset.content,
        "meta": parsed_meta,
        "created_at": asset.created_at.isoformat() if asset.created_at else None
    }


@router.post("/")
def create_studio_asset(
    payload: StudioAssetCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    asset = StudioAsset(
        tool_type=payload.tool_type.strip().lower(),
        title=payload.title.strip(),
        topic=(payload.topic or "").strip() or None,
        platform=(payload.platform or "").strip() or None,
        content=payload.content.strip(),
        meta_json=json.dumps(payload.meta) if payload.meta else None,
        user_id=current_user.id
    )

    db.add(asset)
    db.commit()
    db.refresh(asset)

    return {
        "message": "Studio asset saved successfully",
        "asset": serialize_asset(asset)
    }


@router.get("/")
def list_studio_assets(
    tool_type: Optional[str] = Query(default=None),
    topic: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(StudioAsset).filter(StudioAsset.user_id == current_user.id)

    if tool_type:
        query = query.filter(StudioAsset.tool_type == tool_type.strip().lower())

    if topic:
        query = query.filter(StudioAsset.topic.ilike(f"%{topic.strip()}%"))

    if platform:
        query = query.filter(StudioAsset.platform.ilike(f"%{platform.strip()}%"))

    assets = query.order_by(StudioAsset.created_at.desc()).all()

    return [serialize_asset(asset) for asset in assets]


@router.delete("/{asset_id}")
def delete_studio_asset(
    asset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    asset = (
        db.query(StudioAsset)
        .filter(StudioAsset.id == asset_id, StudioAsset.user_id == current_user.id)
        .first()
    )

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Studio asset not found"
        )

    db.delete(asset)
    db.commit()

    return {
        "message": "Studio asset deleted successfully",
        "asset_id": asset_id
    }