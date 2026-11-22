"""Estado global do modelo (Issues 4 e 6).

O modelo treinado e imutavel depois de criado: a predicao le uma referencia
atomica, e o treino so troca a referencia no final (commit). Assim treino e
predicao nao se bloqueiam, e dois treinos (ou treino + reset) simultaneos sao
recusados com 409 por meio da flag `_training`.
"""
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from app.errors import BusyError
from app.ml.model import LinearModel


@dataclass(frozen=True)
class TrainedModel:
    model: LinearModel
    metrics: dict
    trained_at: str
    n_rows: int


class ModelStore:
    def __init__(self, store_dir: Path):
        self.dir = Path(store_dir)
        self._lock = threading.Lock()
        self._current: TrainedModel | None = None
        self._training = False

    @property
    def model_path(self) -> Path:
        return self.dir / "model.joblib"

    def clear_files(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self) -> TrainedModel | None:
        with self._lock:
            return self._current

    def begin_training(self) -> None:
        with self._lock:
            if self._training:
                raise BusyError("Ja existe um treino em andamento.")
            self._training = True

    def end_training(self) -> None:
        with self._lock:
            self._training = False

    def commit(self, trained: TrainedModel) -> None:
        trained.model.save(self.model_path)
        with self._lock:
            self._current = trained

    def reset(self) -> None:
        with self._lock:
            if self._training:
                raise BusyError("Nao e possivel resetar durante um treino.")
            self._current = None
            self.clear_files()
