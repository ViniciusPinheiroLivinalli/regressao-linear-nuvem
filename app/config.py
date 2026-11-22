"""Configuracao central. Valores podem ser sobrescritos por variaveis de ambiente."""
import os
from pathlib import Path

FEATURES = ["time-5", "time-4", "time-3", "time-2", "time-1"]
TARGET = "time"
TRAIN_COLUMNS = FEATURES + [TARGET]

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 1_048_576))  # 1 MB
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", 30))
CV_SPLITS = int(os.getenv("CV_SPLITS", 5))
MODEL_STORE_DIR = Path(os.getenv("MODEL_STORE_DIR", "model_store"))
APP_VERSION = os.getenv("APP_VERSION", "dev")
