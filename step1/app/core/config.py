from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Mock ML Serving API"
    app_version: str = "0.1.0"
    description: str = "FastAPI 기초 실습 — 가짜 ML 모델 서빙"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
