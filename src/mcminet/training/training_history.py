"""Patient-weighted structured history with explicit internal-validation selection."""
import copy
import json
import os
from pathlib import Path
import tempfile
from mcminet.training.epoch_runner import EpochResult
from mcminet.training.scheduler import ValidationMetric


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrainingHistory:
    def __init__(self):
        self._rows = []

    def append(self, epoch, training, validation, metric, learning_rates):
        if type(epoch) is not int or epoch < 0 or (self._rows and epoch <= self._rows[-1]["epoch"]):
            raise ValueError("History epochs must increase")
        if not isinstance(training, EpochResult) or training.source != "training":
            raise ValueError("Training epoch required")
        if not isinstance(validation, EpochResult) or validation.source != "validation":
            raise ValueError("Internal validation epoch required")
        if not isinstance(metric, ValidationMetric):
            raise ValueError("Validation selection metric required")
        if metric.name in validation.losses and metric.value != validation.losses[metric.name]:
            raise ValueError("Selection metric differs from validation loss")
        import math
        if not learning_rates or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in learning_rates.values()):
            raise ValueError("Invalid learning rates")
        self._rows.append(dict(epoch=epoch, training=training.to_dict(), validation=validation.to_dict(),
                               selection=dict(name=metric.name, value=metric.value, source=metric.source),
                               learning_rates=dict(learning_rates)))

    def state_dict(self):
        return copy.deepcopy(self._rows)

    def load_state_dict(self, rows):
        if not isinstance(rows, list):
            raise ValueError("Invalid history schema")
        candidate = TrainingHistory()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"epoch", "training", "validation", "selection", "learning_rates"}:
                raise ValueError("Invalid history row")
            candidate.append(row["epoch"], EpochResult(**row["training"]), EpochResult(**row["validation"]),
                             ValidationMetric(**row["selection"]), row["learning_rates"])
        self._rows = candidate._rows

    def save_json(self, path, metadata):
        payload = dict(metadata=metadata, history=self.state_dict())
        text = json.dumps(payload, indent=2, allow_nan=False)
        atomic_write(path, lambda p: Path(p).write_text(text, encoding="utf-8"))
