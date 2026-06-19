from datetime import datetime, timezone

import resend
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.hook import Hook
from app.models.referral import Referral
from app.models.user import User
from app.services.referral_service import (
    assign_referral_to_user,
    ensure_user_has_referral_code,
)
from app.services.reward_credit_service import (
    award_signup_verified_reward,
)
from app.utils.plan_limits import (
    get_next_reset_date,
    get_usage_snapshot,
    reset_generation_cycle_if_needed,
)
from app.utils.security import (
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    get_current_user,
    hash_password,
    verify_email_verification_token,
    verify_password,
    verify_password_reset_token,
)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    referral_code: str | None = Field(default=None, max_length=64)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    token: str


class ChangePlanRequest(BaseModel):
    plan: str = Field(..., pattern="^(free|creator|pro)$")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10)
    new_password: str = Field(..., min_length=6, max_length=128)


class ContactSupportRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    category: str = Field(
        ...,
        pattern="^(suggestion|review|complaint|billing|support)$"
    )
    message: str = Field(..., min_length=10, max_length=5000)


class UpdateProfileRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class VerifyEmailRequest(BaseModel):
    token: str = Field(..., min_length=10)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


def get_admin_email_set() -> set[str]:
    raw = settings.admin_emails or ""
    return {
        email.strip().lower()
        for email in raw.split(",")
        if email.strip()
    }


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

    days_until_reset = max(
        0,
        delta.days if delta.seconds == 0 else delta.days + 1
    )

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
        "billing_interval": user.billing_interval or "monthly",
        "billing_status": user.billing_status,
        "created_at": (
            user.created_at.isoformat()
            if user.created_at
            else None
        ),
        "is_admin": is_admin_email(user.email),
        "is_email_verified": bool(user.is_email_verified),
        "referral_code": user.referral_code,
        "reward_credits_balance_naira": (
            user.reward_credits_balance_naira or 0
        ),
        "wallet_balance_naira": (
            user.wallet_balance_naira or 0
        ),
    }


def _ensure_resend_config():
    if not settings.resend_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resend API key is not configured.",
        )

    resend.api_key = settings.resend_api_key.strip()


@router.post("/register")
async def register_user(
    user_data: UserRegister,
    request: Request,
    db: Session = Depends(get_db),
):
    email = user_data.email.lower().strip()
    referral_code = (user_data.referral_code or "").strip()

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = User(
        username=user_data.username.strip(),
        email=email,
        hashed_password=hash_password(
            user_data.password
        ),
        plan="free",
        billing_interval="monthly",
        monthly_generation_count=0,
        generation_reset_date=get_next_reset_date(),
        billing_status="inactive",
        is_email_verified=0,
        reward_credits_balance_naira=0,
        wallet_balance_naira=0,
    )

    db.add(new_user)
    db.flush()

    try:
        ensure_user_has_referral_code(
            db,
            new_user
        )

        if referral_code:
            try:
                assign_referral_to_user(
                    db,
                    new_user,
                    referral_code
                )
            except ValueError as exc:
                db.rollback()

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                )

        db.commit()
        db.refresh(new_user)

    except HTTPException:
        raise

    except Exception:
        db.rollback()
        raise

    return {
        "message": (
            "Registration successful. "
            "Please verify your email."
        ),
        "user": serialize_basic_user(new_user),
    }


@router.post("/login")
def login_user(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    email = credentials.email.lower().strip()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if (
        not user
        or not verify_password(
            credentials.password,
            user.hashed_password
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    reset_generation_cycle_if_needed(user)

    ensure_user_has_referral_code(
        db,
        user
    )

    db.commit()
    db.refresh(user)

    access_token = create_access_token({
        "sub": str(user.id)
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_basic_user(user),
    }


@router.post("/google-login")
def google_login(
    payload: GoogleLoginRequest,
    db: Session = Depends(get_db),
):
    try:
        decoded_token = firebase_auth.verify_id_token(
            payload.token
        )

        email = decoded_token.get("email")
        name = decoded_token.get("name")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account email missing",
            )

        email = email.lower().strip()

        user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        # Auto-create Google users
        if not user:
            username = (
                name.lower().replace(" ", "_")
                if name
                else email.split("@")[0]
            )

            user = User(
                username=username,
                email=email,
                hashed_password="GOOGLE_AUTH",
                plan="free",
                billing_interval="monthly",
                monthly_generation_count=0,
                generation_reset_date=get_next_reset_date(),
                billing_status="inactive",
                is_email_verified=1,
                reward_credits_balance_naira=0,
                wallet_balance_naira=0,
            )

            db.add(user)
            db.flush()

            ensure_user_has_referral_code(
                db,
                user
            )

            db.commit()
            db.refresh(user)

        reset_generation_cycle_if_needed(user)

        access_token = create_access_token({
            "sub": str(user.id)
        })

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": serialize_basic_user(user),
        }

    except Exception as e:
        print("GOOGLE LOGIN ERROR:", e)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        )


@router.post("/token")
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    email = form_data.username.lower().strip()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if (
        not user
        or not verify_password(
            form_data.password,
            user.hashed_password
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = create_access_token({
        "sub": str(user.id)
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }

@router.get("/profile")
def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    reset_generation_cycle_if_needed(current_user)

    saved_hooks_count = (
    db.query(Hook)
    .filter(Hook.user_id == current_user.id)
    .count()
)

    usage = get_usage_snapshot(
    current_user,
    saved_hooks_count
)
    db.commit()
    db.refresh(current_user)

    return {
        **serialize_basic_user(current_user),
        "usage": usage,
        "cycle": build_cycle_info(current_user),
    }