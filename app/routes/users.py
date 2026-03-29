from datetime import datetime, timezone

import resend
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.hook import Hook
from app.models.referral import Referral
from app.models.user import User
from app.services.referral_service import assign_referral_to_user, ensure_user_has_referral_code
from app.services.reward_credit_service import award_signup_verified_reward
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
    category: str = Field(..., pattern="^(suggestion|review|complaint|billing|support)$")
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
        "billing_interval": user.billing_interval or "monthly",
        "billing_status": user.billing_status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "is_admin": is_admin_email(user.email),
        "is_email_verified": bool(user.is_email_verified),
        "referral_code": user.referral_code,
        "reward_credits_balance_naira": user.reward_credits_balance_naira or 0,
        "wallet_balance_naira": user.wallet_balance_naira or 0,
    }


def _ensure_resend_config():
    if not settings.resend_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resend API key is not configured.",
        )

    resend.api_key = settings.resend_api_key.strip()


def send_password_reset_email(to_email: str, reset_link: str):
    _ensure_resend_config()

    html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111827;">
      <h2>Reset your Hookora password</h2>
      <p>We received a request to reset the password for your Hookora account.</p>
      <p>
        <a href="{reset_link}" style="display:inline-block;padding:12px 18px;background:#2563eb;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:700;">
          Reset Password
        </a>
      </p>
      <p>If the button above does not work, copy and paste this link into your browser:</p>
      <p>{reset_link}</p>
      <p>This link expires in 30 minutes.</p>
      <p>If you did not request this, you can safely ignore this email.</p>
    </div>
    """

    resend.Emails.send({
        "from": settings.email_from.strip(),
        "to": [to_email],
        "subject": "Reset your Hookora password",
        "html": html,
    })


def send_email_verification_email(to_email: str, verify_link: str):
    _ensure_resend_config()

    html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111827;">
      <h2>Verify your Hookora email</h2>
      <p>Welcome to Hookora. Please verify your email address to unlock your referral eligibility and secure your account.</p>
      <p>
        <a href="{verify_link}" style="display:inline-block;padding:12px 18px;background:#111827;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:700;">
          Verify Email
        </a>
      </p>
      <p>If the button above does not work, copy and paste this link into your browser:</p>
      <p>{verify_link}</p>
      <p>This verification link expires in 24 hours.</p>
    </div>
    """

    resend.Emails.send({
        "from": settings.email_from.strip(),
        "to": [to_email],
        "subject": "Verify your Hookora email",
        "html": html,
    })


def send_support_email(name: str, email: str, category: str, message: str):
    _ensure_resend_config()

    support_html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111827;">
      <h2>New Hookora support message</h2>
      <p><strong>Name:</strong> {name}</p>
      <p><strong>Email:</strong> {email}</p>
      <p><strong>Category:</strong> {category}</p>
      <p><strong>Message:</strong></p>
      <div style="padding:12px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;white-space:pre-wrap;">{message}</div>
    </div>
    """

    confirmation_html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111827;">
      <h2>We received your message</h2>
      <p>Hi {name},</p>
      <p>Thanks for contacting Hookora. We received your message and will review it as soon as possible.</p>
      <p><strong>Category:</strong> {category}</p>
      <p><strong>Your message:</strong></p>
      <div style="padding:12px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;white-space:pre-wrap;">{message}</div>
      <p>You can reply to this email if you need to add more details.</p>
    </div>
    """

    resend.Emails.send({
        "from": settings.email_from.strip(),
        "to": [settings.support_email.strip()],
        "reply_to": email,
        "subject": f"Hookora Support: {category.title()} from {name}",
        "html": support_html,
    })

    resend.Emails.send({
        "from": settings.email_from.strip(),
        "to": [email],
        "reply_to": settings.support_email.strip(),
        "subject": "We received your Hookora message",
        "html": confirmation_html,
    })


async def broadcast_milestone_update(request: Request, db: Session):
    if not hasattr(request.app.state, "milestone_manager"):
        return

    total_users = db.query(User).count()
    manager = request.app.state.milestone_manager
    await manager.broadcast({
        "type": "milestone_update",
        "total_users": total_users,
    })


def _build_email_verification_link(email: str) -> str:
    token = create_email_verification_token(email)
    return f"{settings.frontend_app_url.rstrip('/')}/verify-email?token={token}"


@router.post("/register")
async def register_user(
    user_data: UserRegister,
    request: Request,
    db: Session = Depends(get_db),
):
    email = user_data.email.lower().strip()
    referral_code = (user_data.referral_code or "").strip()

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
        ensure_user_has_referral_code(db, new_user)

        if referral_code:
            try:
                assign_referral_to_user(db, new_user, referral_code)
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

    verify_link = _build_email_verification_link(new_user.email)
    try:
        send_email_verification_email(new_user.email, verify_link)
    except Exception as exc:
        print(f"[HOOKORA VERIFY EMAIL ERROR] email={new_user.email} error={exc}")

    await broadcast_milestone_update(request, db)

    return {
        "message": "Registration successful. Please verify your email.",
        "user": serialize_basic_user(new_user),
    }


@router.post("/verify-email")
def verify_email(
    payload: VerifyEmailRequest,
    db: Session = Depends(get_db),
):
    email = verify_email_verification_token(payload.token)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    if not int(user.is_email_verified or 0):
        user.is_email_verified = 1
        user.email_verified_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

    referral = db.query(Referral).filter(Referral.referred_user_id == user.id).first()
    if referral and referral.signup_reward_awarded_at is None:
        referral.status = "email_verified"
        db.commit()
        db.refresh(referral)

    award_signup_verified_reward(db, user.id)
    db.refresh(user)

    return {
        "message": "Email verified successfully.",
        "user": serialize_basic_user(user),
    }


@router.post("/resend-verification")
def resend_verification_email(
    payload: ResendVerificationRequest,
    db: Session = Depends(get_db),
):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return {"message": "If an account exists for that email, a new verification email has been sent."}

    if int(user.is_email_verified or 0):
        return {"message": "This email is already verified."}

    verify_link = _build_email_verification_link(user.email)

    try:
        send_email_verification_email(user.email, verify_link)
    except Exception as exc:
        print(f"[HOOKORA RESEND VERIFY EMAIL ERROR] email={user.email} error={exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email.",
        )

    return {"message": "Verification email sent successfully."}


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
    ensure_user_has_referral_code(db, user)
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
    ensure_user_has_referral_code(db, user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if user:
        reset_token = create_password_reset_token(user.email)
        reset_link = (
            f"{settings.frontend_app_url.rstrip('/')}/reset-password?token={reset_token}"
        )

        try:
            send_password_reset_email(user.email, reset_link)
        except Exception as exc:
            print(f"[HOOKORA PASSWORD RESET EMAIL ERROR] email={user.email} error={exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send password reset email.",
            )

    return {
        "message": "If an account exists for that email, a password reset link has been sent."
    }


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = verify_password_reset_token(payload.token)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )

    user.hashed_password = hash_password(payload.new_password)
    db.commit()
    db.refresh(user)

    return {"message": "Password reset successful. You can now log in."}


@router.post("/contact-support")
def contact_support(payload: ContactSupportRequest):
    try:
        send_support_email(
            name=payload.name.strip(),
            email=payload.email.lower().strip(),
            category=payload.category.strip(),
            message=payload.message.strip(),
        )
    except Exception as exc:
        print(f"[HOOKORA SUPPORT EMAIL ERROR] email={payload.email} error={exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send support message. Please try again.",
        )

    return {"message": "Your message has been sent successfully."}


@router.get("/profile")
def get_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reset_generation_cycle_if_needed(current_user)
    ensure_user_has_referral_code(db, current_user)
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
        "billing_interval": current_user.billing_interval or "monthly",
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
        "is_email_verified": bool(current_user.is_email_verified),
        "email_verified_at": (
            current_user.email_verified_at.isoformat()
            if current_user.email_verified_at
            else None
        ),
        "referral_code": current_user.referral_code,
        "referred_by_user_id": current_user.referred_by_user_id,
        "reward_credits_balance_naira": current_user.reward_credits_balance_naira or 0,
        "wallet_balance_naira": current_user.wallet_balance_naira or 0,
        "usage": usage,
        "cycle": cycle,
    }


@router.patch("/profile")
def update_profile(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.username = payload.username.strip()
    db.commit()
    db.refresh(current_user)

    return {
        "message": "Profile updated successfully.",
        "user": serialize_basic_user(current_user),
    }


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current password.",
        )

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()
    db.refresh(current_user)

    return {"message": "Password changed successfully."}


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
    current_user.billing_interval = "monthly"
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