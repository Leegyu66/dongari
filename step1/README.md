# Step 1 — FastAPI Mock ML Serving

저번 주에 배운 네트워크 기초(IP, 포트, 엔드포인트)를 직접 코드로 만들어보는 실습 프로젝트입니다.
무거운 ML 모델 대신 **Mock(가짜) 모델**을 서빙하여, API 통신의 전체 흐름을 먼저 체험하는 것이 목표입니다.

## 실행 방법

```bash
# 1. 가상환경 생성 & 활성화
python -m venv .venv
source .venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
python -m app.main
```

브라우저에서 <http://localhost:8000/docs> 접속 → Swagger UI 및 Postman 에서 직접 테스트!

> 💡 `0.0.0.0`과 `8000` — 저번 주에 배운 그 IP와 포트입니다.

## 폴더 구조

```
step1/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 진입점 (uvicorn이 바라보는 곳)
│   ├── core/
│   │   └── config.py        # 환경 설정
│   ├── schemas/
│   │   └── prediction.py    # Pydantic 스키마 (입출력 데이터 검증)
│   ├── services/
│   │   └── ml_model.py      # Mock ML 모델 로직
│   └── routers/
│       ├── health.py        # GET 예제 (서버 상태 확인)
│       └── predict.py       # POST 예제 (예측 요청)
└── requirements.txt
```

## 학습 포인트

| 구간 | 배우는 것 |
|------|----------|
| `main.py` | FastAPI 앱 생성, 라우터 등록 |
| `routers/health.py` | **GET** 메서드 — 정보를 달라고 요청 |
| `routers/predict.py` | **POST** 메서드 — 데이터를 주고 예측받기 |
| `schemas/prediction.py` | **Pydantic** — 타입 강제 & 자동 검증 |
| `services/ml_model.py` | 비즈니스 로직 분리 (라우터는 얇게!) |
