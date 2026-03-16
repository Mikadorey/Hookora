from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.hook import Hook
from app.models.user import User
from app.utils.plan_limits import (
    get_next_reset_date,
    get_usage_snapshot,
    reset_generation_cycle_if_needed,
)
from app.utils.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class ChangePlanRequest(BaseModel):
    plan: str = Field(..., pattern="^(free|creator|pro)$")


def get_admin_email_set() -> set[str]:
    raw = settings.admin_emails or ""
    return {email.strip().lower() for email in raw.split(",") if email.strip()}


def is_admin_email(email: str) -> bool:
    return email.lower().strip() in get_admin_email_set()


def build_cycle_info(user: User) -> dict:
    if not user.generation_reset_date:
        return {
            "reset_date": None,
            "days_until_reset": None,
            "reset_soon": False,
        }

    reset_date = user.generation_reset_date
    now = datetime.now(timezone.utc)

    if reset_date.tzinfo is None:
        reset_date = reset_date.replace(tzinfo=timezone.utc)

    delta = reset_date - now
    days_until_reset = max(0, delta.days if delta.seconds == 0 else delta.days + 1)

    return {
        "reset_date": reset_date.isoformat(),
        "days_until_reset": days_until_reset,
        "reset_soon": days_until_reset <= 3,
    }


def serialize_basic_user(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "plan": user.plan,
        "billing_status": user.billing_status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "is_admin": is_admin_email(user.email),
    }


@router.post("/register")
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    email = user_data.email.lower().strip()

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = User(
        username=user_data.username.strip(),
        email=email,
        hashed_password=hash_password(user_data.password),
        plan="free",
        monthly_generation_count=0,
        generation_reset_date=get_next_reset_date(),
        billing_status="inactive",
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return serialize_basic_user(new_user)


@router.post("/login")
def login_user(credentials: UserLogin, db: Session = Depends(get_db)):
    email = credentials.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    reset_generation_cycle_if_needed(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_basic_user(user),
    }


@router.post("/token")
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    email = form_data.username.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    reset_generation_cycle_if_needed(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/profile")
def get_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reset_generation_cycle_if_needed(current_user)
    db.commit()
    db.refresh(current_user)

    saved_hooks_count = db.query(Hook).filter(Hook.user_id == current_user.id).count()
    usage = get_usage_snapshot(current_user, saved_hooks_count)
    cycle = build_cycle_info(current_user)

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "plan": current_user.plan,
        "billing_status": current_user.billing_status,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "paystack_customer_code": current_user.paystack_customer_code,
        "paystack_subscription_code": current_user.paystack_subscription_code,
        "paystack_plan_code": current_user.paystack_plan_code,
        "billing_current_period_end": (
            current_user.billing_current_period_end.isoformat()
            if current_user.billing_current_period_end
            else None
        ),
        "is_admin": is_admin_email(current_user.email),
        "usage": usage,
        "cycle": cycle,
    }


@router.post("/change-plan")
def change_plan(
    payload: ChangePlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    requested_plan = payload.plan.lower().strip()

    if requested_plan not in {"free", "creator", "pro"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid plan selected",
        )

    if requested_plan in {"creator", "pro"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Paid plans must be activated through billing checkout.",
        )

    current_user.plan = "free"
    current_user.billing_status = "inactive"
    current_user.paystack_subscription_code = None
    current_user.paystack_plan_code = None
    current_user.billing_current_period_end = None

    reset_generation_cycle_if_needed(current_user)
    db.commit()
    db.refresh(current_user)

    saved_hooks_count = db.query(Hook).filter(Hook.user_id == current_user.id).count()
    usage = get_usage_snapshot(current_user, saved_hooks_count)
    cycle = build_cycle_info(current_user)

    return {
        "message": "Plan updated to free",
        "user": serialize_basic_user(current_user),
        "usage": usage,
        "cycle": cycle,
    }