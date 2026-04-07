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


TOOL_TYPE_ALIASES = {
    "hooks": "hooks",
    "hook_generator": "hooks",
    "viral_hooks": "hooks",
    "viral_hook_generator": "hooks",

    "youtube_title_optimizer": "youtube_title_optimizer",
    "youtube_titles": "youtube_title_optimizer",
    "youtube_title": "youtube_title_optimizer",

    "caption_generator": "caption_generator",
    "captions": "caption_generator",
    "caption": "caption_generator",

    "hashtag_generator": "hashtag_generator",
    "hashtags": "hashtag_generator",
    "hashtag": "hashtag_generator",

    "description_rewriter": "description_rewriter",
    "description_rewrite": "description_rewriter",
    "descriptions": "description_rewriter",
    "description": "description_rewriter",

    "short_script_generator": "short_script_generator",
    "short_scripts": "short_script_generator",
    "short_script": "short_script_generator",

    "thumbnail_text_analyzer": "thumbnail_text_analyzer",
    "thumbnail_text": "thumbnail_text_analyzer",
    "thumbnail_texts": "thumbnail_text_analyzer",

    "hook_angles": "hook_angles",
    "hook_angle": "hook_angles",

    "cta_generator": "cta_generator",
    "ctas": "cta_generator",
    "cta": "cta_generator",

    "humanizer": "humanizer",
    "ai_humanizer": "humanizer",
    "ai_humanizer_tool": "humanizer",
    "humanized_rewrite": "humanizer",
    "humanized_rewrites": "humanizer",
}

TOOL_META = {
    "hooks": {
        "label": "Viral Hook Generator",
        "badge": "HOOKS",
    },
    "youtube_title_optimizer": {
        "label": "YouTube Title Optimizer",
        "badge": "TITLE",
    },
    "caption_generator": {
        "label": "Caption Generator",
        "badge": "CAPTION",
    },
    "hashtag_generator": {
        "label": "Hashtag Generator",
        "badge": "HASHTAG",
    },
    "description_rewriter": {
        "label": "Description Rewriter",
        "badge": "DESCRIPTION",
    },
    "short_script_generator": {
        "label": "Short Script Generator",
        "badge": "SCRIPT",
    },
    "thumbnail_text_analyzer": {
        "label": "Thumbnail Text Analyzer",
        "badge": "THUMBNAIL",
    },
    "hook_angles": {
        "label": "Hook Angles",
        "badge": "ANGLES",
    },
    "cta_generator": {
        "label": "CTA Generator",
        "badge": "CTA",
    },
    "humanizer": {
        "label": "AI Humanizer",
        "badge": "HUMANIZER",
    },
    "other": {
        "label": "Other",
        "badge": "OTHER",
    },
}


class StudioAssetCreateRequest(BaseModel):
    tool_type: str = Field(..., min_length=2, max_length=50)
    title: str = Field(..., min_length=2, max_length=200)
    topic: Optional[str] = Field(default=None, max_length=200)
    platform: Optional[str] = Field(default=None, max_length=50)
    content: str = Field(..., min_length=1)
    meta: Optional[dict] = None


def normalize_tool_type(value: Optional[str]) -> str:
    if not value:
        return "other"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return TOOL_TYPE_ALIASES.get(normalized, normalized if normalized in TOOL_META else "other")


def get_tool_meta(tool_type: Optional[str]) -> dict:
    normalized = normalize_tool_type(tool_type)
    return TOOL_META.get(normalized, TOOL_META["other"])


def parse_asset_meta(asset: StudioAsset):
    parsed_meta = None

    if asset.meta_json:
        try:
            parsed_meta = json.loads(asset.meta_json)
        except json.JSONDecodeError:
            parsed_meta = None

    return parsed_meta


def serialize_asset(asset: StudioAsset) -> dict:
    normalized_tool_type = normalize_tool_type(asset.tool_type)
    meta = get_tool_meta(normalized_tool_type)
    parsed_meta = parse_asset_meta(asset)

    return {
        "id": asset.id,
        "tool_type": normalized_tool_type,
        "tool_label": meta["label"],
        "tool_badge": meta["badge"],
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
    normalized_tool_type = normalize_tool_type(payload.tool_type)

    asset = StudioAsset(
        tool_type=normalized_tool_type,
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

    assets = query.order_by(StudioAsset.created_at.desc()).all()

    normalized_filter_tool_type = normalize_tool_type(tool_type) if tool_type else None
    normalized_topic = topic.strip().lower() if topic else None
    normalized_platform = platform.strip().lower() if platform else None

    filtered_assets = []

    for asset in assets:
        normalized_asset_tool_type = normalize_tool_type(asset.tool_type)

        if normalized_filter_tool_type and normalized_asset_tool_type != normalized_filter_tool_type:
            continue

        asset_topic = (asset.topic or "").strip().lower()
        if normalized_topic and normalized_topic not in asset_topic:
            continue

        asset_platform = (asset.platform or "").strip().lower()
        if normalized_platform and normalized_platform not in asset_platform:
            continue

        filtered_assets.append(asset)

    return [serialize_asset(asset) for asset in filtered_assets]


@router.get("/meta")
def get_studio_asset_meta(
    current_user: User = Depends(get_current_user),
):
    return {
        "tool_types": [
            {
                "tool_type": tool_type,
                "tool_label": meta["label"],
                "tool_badge": meta["badge"],
            }
            for tool_type, meta in TOOL_META.items()
            if tool_type != "other"
        ]
    }


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