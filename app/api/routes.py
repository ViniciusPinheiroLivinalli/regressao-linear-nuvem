"""Rotas HTTP.

Tratamento ponta-a-ponta: todo arquivo trafega compactado com Huffman Canonico
(app/huffman.py no servidor, app/web/huffman.js no navegador). O upload chega
compactado e e descompactado so em memoria; o CSV de previsoes volta compactado
(base64 dentro do JSON). As metricas voltam como JSON simples, pois nao sao arquivos.
"""
import base64

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config, huffman
from app.errors import CompressionError, PayloadTooLargeError
from app.ml import service

router = APIRouter()


def _too_large() -> PayloadTooLargeError:
    return PayloadTooLargeError(
        f"Arquivo maior que o limite de {config.MAX_UPLOAD_BYTES} bytes."
    )


def _decode(payload: bytes) -> bytes:
    """Descompacta o upload, recusando antes se o tamanho declarado passar do limite."""
    try:
        tamanho = huffman.validar_cabecalho(payload)
        if tamanho > config.MAX_UPLOAD_BYTES:
            raise _too_large()
        return huffman.descompactar(payload)
    except ValueError as exc:
        raise CompressionError(f"Arquivo compactado invalido: {exc}")


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise _too_large()
    return await run_in_threadpool(_decode, payload)


@router.get("/health")
def health():
    return {"status": "ok", "version": config.APP_VERSION}


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
    metrics, training, warnings = await run_in_threadpool(
        service.train, request.app.state.store, raw
    )
    body = {**metrics, "training": training}
    if warnings:
        body["warnings"] = warnings
    return body


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await _read_upload(file)
    result = await run_in_threadpool(service.predict, request.app.state.store, raw)
    packed = await run_in_threadpool(huffman.compactar, result.csv.encode("utf-8"))
    return {
        "labeled": result.labeled,
        "n_rows": result.n_rows,
        "metrics": result.metrics,
        "expected_metrics": result.expected_metrics,
        "predictions_huff": base64.b64encode(packed).decode("ascii"),
        "warnings": result.warnings,
    }


@router.post("/reset")
def reset(request: Request):
    request.app.state.store.reset()
    return {"trained": False}
