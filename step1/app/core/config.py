from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Trump War Predictor API"
    app_version: str = "0.2.0"
    description: str = "🇺🇸 트럼프 전쟁 확률 예측 API — FastAPI + PostgreSQL 실습"
    database_url: str = "postgresql://trump:bigmac123@localhost:5432/trump_war"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
