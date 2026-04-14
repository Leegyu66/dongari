from fastapi import APIRouter, status

from app.schemas.prediction import TrumpInput, WarPredictionResponse
from app.services.ml_model import ml_model

router = APIRouter(tags=["predict"])


@router.post(
    "/",
    response_model=WarPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="🇺🇸 트럼프 전쟁 확률 예측",
    description="트럼프 스트레스 지수, 빅맥 섭취량, 수면시간을 받아 전쟁 발발 시나리오를 예측합니다.",
)
def predict(user_input: TrumpInput) -> WarPredictionResponse:
    return ml_model.predict(user_input)
