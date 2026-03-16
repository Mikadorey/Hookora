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


def init_db():
    from app.models.hook import Hook  # noqa: F401
    from app.models.studio_asset import StudioAsset  # noqa: F401
    from app.models.studio_usage_event import StudioUsageEvent  # noqa: F401
    from app.models.user import User  # noqa: F401
    from app.models.waitlist_interest import WaitlistInterest  # noqa: F401

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

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
                        "ALTER TABLE users ADD COLUMN generation_reset_date DATETIME"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN created_at DATETIME"
                    )
                )
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
                        "ALTER TABLE users ADD COLUMN billing_current_period_end DATETIME"
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()