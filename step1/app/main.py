from fastapi import FastAPI

from app.core.config import settings
from app.routers import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=settings.description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(api_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)