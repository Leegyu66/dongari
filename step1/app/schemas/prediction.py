from pydantic import BaseModel, Field
from typing import Literal


class TrumpInput(BaseModel):
    """트럼프 전쟁 확률 예측에 사용할 피처."""

    trump_stress_index: int = Field(
        ..., ge=0, le=100, description="트럼프 스트레스 지수 (0~100)"
    )
    big_mac_count: int = Field(
        ..., ge=0, le=10, description="오늘 빅맥 섭취량 (0~10개)"
    )
    sleep_hours: float = Field(
        ..., ge=0.0, le=24.0, description="어젯밤 수면 시간 (0~24)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"trump_stress_index": 85, "big_mac_count": 6, "sleep_hours": 3.5}
            ]
        }
    }


class WarPredictionResponse(BaseModel):
    """전쟁 확률 예측 응답 스키마."""

    label: Literal[
        "peaceful",
        "tariff_war",
        "trade_war",
        "diplomatic_crisis",
        "limited_strike",
        "world_war_3",
    ] = Field(..., description="예측된 시나리오")
    probability: float = Field(..., ge=0.0, le=1.0, description="전쟁 발발 확률")
    reason: str = Field(..., description="이 결과가 나온 이유 (한 줄 풍자)")
    model_version: str = Field(..., description="사용된 모델 버전")
