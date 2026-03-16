from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    jwt_secret: str
    admin_emails: str = ""

    paystack_secret_key: str = ""
    paystack_public_key: str = ""
    paystack_plan_creator_monthly: str = ""
    paystack_plan_pro_monthly: str = ""

    frontend_app_url: str = "https://hookora.app"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()