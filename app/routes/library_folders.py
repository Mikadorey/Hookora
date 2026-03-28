from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.library_folder import LibraryFolder
from app.models.user import User
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/library-folders",
    tags=["Library Folders"],
)


class UpsertLibraryFolderRequest(BaseModel):
    folder_key: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)


def serialize_folder(folder: LibraryFolder) -> dict:
    return {
        "id": folder.id,
        "folder_key": folder.folder_key,
        "display_name": folder.display_name,
        "created_at": folder.created_at.isoformat() if folder.created_at else None,
        "updated_at": folder.updated_at.isoformat() if folder.updated_at else None,
    }


@router.get("/")
def get_library_folders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    folders = (
        db.query(LibraryFolder)
        .filter(LibraryFolder.user_id == current_user.id)
        .order_by(LibraryFolder.updated_at.desc(), LibraryFolder.id.desc())
        .all()
    )

    return [serialize_folder(folder) for folder in folders]


@router.post("/upsert")
def upsert_library_folder(
    payload: UpsertLibraryFolderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    folder_key = payload.folder_key.strip().lower()
    display_name = payload.display_name.strip()

    if not folder_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder key is required.",
        )

    if not display_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Display name is required.",
        )

    folder = (
        db.query(LibraryFolder)
        .filter(
            LibraryFolder.user_id == current_user.id,
            LibraryFolder.folder_key == folder_key,
        )
        .first()
    )

    if folder:
        folder.display_name = display_name
    else:
        folder = LibraryFolder(
            user_id=current_user.id,
            folder_key=folder_key,
            display_name=display_name,
        )
        db.add(folder)

    db.commit()
    db.refresh(folder)

    return {
        "message": "Folder updated successfully.",
        "folder": serialize_folder(folder),
    }


@router.delete("/{folder_key}")
def delete_library_folder_name(
    folder_key: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    normalized_key = (folder_key or "").strip().lower()
    if not normalized_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder key is required.",
        )

    folder = (
        db.query(LibraryFolder)
        .filter(
            LibraryFolder.user_id == current_user.id,
            LibraryFolder.folder_key == normalized_key,
        )
        .first()
    )

    if folder:
        db.delete(folder)
        db.commit()

    return {"message": "Folder name removed successfully."}