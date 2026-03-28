from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

SQLALCHEMY_DATABASE_URL = settings.database_url.strip()

connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _datetime_type() -> str:
    if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
        return "TIMESTAMP"
    return "DATETIME"


def _json_type() -> str:
    if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
        return "JSONB"
    return "JSON"


def init_db():
    from app.models.hook import Hook  # noqa: F401
    from app.models.library_folder import LibraryFolder  # noqa: F401
    from app.models.referral import Referral  # noqa: F401
    from app.models.reward_credit_ledger import RewardCreditLedger  # noqa: F401
    from app.models.wallet_ledger import WalletLedger  # noqa: F401
    from app.models.studio_asset import StudioAsset  # noqa: F401
    from app.models.studio_usage_event import StudioUsageEvent  # noqa: F401
    from app.models.user import User  # noqa: F401
    from app.models.waitlist_interest import WaitlistInterest  # noqa: F401

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    dt_type = _datetime_type()

    if "hooks" in tables:
        columns = [column["name"] for column in inspector.get_columns("hooks")]

        if "is_favorite" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE hooks ADD COLUMN is_favorite BOOLEAN NOT NULL DEFAULT 0"
                    )
                )

    if "users" in tables:
        columns = [column["name"] for column in inspector.get_columns("users")]

        if "plan" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN plan VARCHAR NOT NULL DEFAULT 'free'"
                    )
                )

        if "monthly_generation_count" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN monthly_generation_count INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "generation_reset_date" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN generation_reset_date {dt_type}"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN created_at {dt_type}"
                    )
                )
                if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
                    connection.execute(
                        text(
                            "UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                        )
                    )
                else:
                    connection.execute(
                        text(
                            "UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                        )
                    )

        if "billing_status" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN billing_status VARCHAR NOT NULL DEFAULT 'inactive'"
                    )
                )

        if "billing_current_period_end" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN billing_current_period_end {dt_type}"
                    )
                )

        if "paystack_customer_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_customer_code VARCHAR"
                    )
                )

        if "paystack_subscription_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_subscription_code VARCHAR"
                    )
                )

        if "paystack_plan_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_plan_code VARCHAR"
                    )
                )

        if "is_email_verified" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN is_email_verified INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "email_verified_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN email_verified_at {dt_type}"
                    )
                )

        if "referral_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN referral_code VARCHAR"
                    )
                )

        if "referred_by_user_id" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN referred_by_user_id INTEGER"
                    )
                )

        if "referral_source_locked_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN referral_source_locked_at {dt_type}"
                    )
                )

        if "reward_credits_balance_naira" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN reward_credits_balance_naira INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "wallet_balance_naira" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN wallet_balance_naira INTEGER NOT NULL DEFAULT 0"
                    )
                )

        columns = [column["name"] for column in inspector.get_columns("users")]

        if "referral_code" in columns:
            with engine.begin() as connection:
                rows = connection.execute(
                    text("SELECT id FROM users WHERE referral_code IS NULL OR referral_code = ''")
                ).fetchall()

                for row in rows:
                    user_id = int(row[0])
                    referral_code = f"HKR{user_id:06d}"
                    connection.execute(
                        text(
                            "UPDATE users SET referral_code = :referral_code WHERE id = :user_id"
                        ),
                        {"referral_code": referral_code, "user_id": user_id},
                    )

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if "library_folders" not in tables:
        Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()