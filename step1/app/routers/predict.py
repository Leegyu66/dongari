from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.prediction import PredictionLog
from app.schemas.prediction import (
    PredictionListResponse,
    TrumpInput,
    WarPredictionResponse,
)
from app.services.ml_model import ml_model

router = APIRouter(tags=["predict"])


@router.post(
    "/",
    response_model=WarPredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="🇺🇸 트럼프 전쟁 확률 예측",
    description="스트레스 지수, 빅맥 섭취량, 수면시간을 받아 전쟁 확률을 예측하고 DB에 저장합니다.",
)
def predict(user_input: TrumpInput, db: Session = Depends(get_db)):
    result = ml_model.predict(user_input)

    log = PredictionLog(
        trump_stress_index=user_input.trump_stress_index,
        big_mac_count=user_input.big_mac_count,
        sleep_hours=user_input.sleep_hours,
        label=result.label,
        probability=result.probability,
        reason=result.reason,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return WarPredictionResponse(
        id=log.id,
        label=log.label,
        probability=log.probability,
        reason=log.reason,
        model_version=result.model_version,
        created_at=log.created_at,
    )


@router.get(
    "/",
    response_model=PredictionListResponse,
    summary="📋 예측 기록 조회",
    description="저장된 예측 기록을 최신순으로 조회합니다.",
)
def get_predictions(
    skip: int = Query(0, ge=0, description="건너뛸 개수"),
    limit: int = Query(20, ge=1, le=100, description="조회할 개수 (최대 100)"),
    db: Session = Depends(get_db),
):
    total = db.query(PredictionLog).count()
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return PredictionListResponse(
        total=total,
        predictions=[
            WarPredictionResponse(
                id=log.id,
                label=log.label,
                probability=log.probability,
                reason=log.reason,
                model_version=ml_model.VERSION,
                created_at=log.created_at,
            )
            for log in logs
        ],
    )
