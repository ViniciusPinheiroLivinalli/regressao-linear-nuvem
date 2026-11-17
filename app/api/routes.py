"""Issues 1, 4, 5 e 6: rotas HTTP.

Nesta fase os arquivos chegam em claro (multipart). A camada de compactacao e
criptografia (Issues 7 e 8) vai envolver `raw` na entrada e `csv` na saida.
"""
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config
from app.errors import PayloadTooLargeError
from app.ml import service

router = APIRouter()


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Arquivo maior que o limite de {config.MAX_UPLOAD_BYTES} bytes."
        )
    return raw


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status(request: Request):
    trained = request.app.state.store.get()
    if trained is None:
        return {"trained": False, "trained_at": None, "n_rows": None, "metrics": None}
    return {
        "trained": True,
        "trained_at": trained.trained_at,
        "n_rows": trained.n_rows,
        "metrics": trained.metrics,
    }


@router.post("/train")
async def train(request: Request, file: UploadFile = File(...)):
    raw = await _read_upload(file)
    return await run_in_threadpool(service.train, request.app.state.store, raw)


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await _read_upload(file)
    result = await run_in_threadpool(service.predict, request.app.state.store, raw)
    return {
        "labeled": result.labeled,
        "n_rows": result.n_rows,
        "metrics": result.metrics,
        "predictions_csv": result.csv,
    }


@router.post("/reset")
def reset(request: Request):
    request.app.state.store.reset()
    return {"trained": False}
