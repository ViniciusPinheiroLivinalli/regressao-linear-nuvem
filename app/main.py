from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes import router
from app.errors import AppError
from app.state import ModelStore

WEB_DIR = Path(__file__).parent / "web"


def create_app(store_dir: Path | None = None) -> FastAPI:
    store = ModelStore(store_dir or config.MODEL_STORE_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.clear_files()  # todo start comeca "sem modelo treinado"
        yield

    app = FastAPI(title="Regressao Linear na Nuvem", lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    app.include_router(router)
    # Registrado por ultimo para nao sombrear as rotas da API.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
