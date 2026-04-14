from pydantic import BaseModel
from typing import Literal

class HealthResponse(BaseModel):
    """헬스체크 응답 스키마."""

    status: Literal["ok"]
    app_name: str
    version: str