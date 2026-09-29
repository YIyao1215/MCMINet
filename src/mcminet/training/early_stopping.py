"""Validation-only early-stopping state; no implicit model selection source."""
from mcminet.config import default
from dataclasses import dataclass, asdict
import math
from mcminet.training.scheduler import ValidationMetric
from mcminet.training.training_config import finite_number


@dataclass
class EarlyStopping:
    mode: str
    patience: int
    min_delta: float
    metric_name: str = default("early_stopping.monitor")
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    last_epoch: int = -1

    def __post_init__(self):
        if self.mode not in ("min", "max") or type(self.patience) is not int or self.patience < 1:
            raise ValueError("Invalid early-stopping policy")
        finite_number("min_delta", self.min_delta)
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("Metric name required")
        if type(self.last_epoch) is not int or self.last_epoch < -1 or type(self.bad_epochs) is not int or self.bad_epochs < 0:
            raise ValueError("Invalid early-stopping counters")
        if self.best_value is None:
            if self.best_epoch is not None or self.last_epoch != -1 or self.bad_epochs != 0:
                raise ValueError("Invalid uninitialized early-stopping state")
        elif (isinstance(self.best_value, bool) or not isinstance(self.best_value, (int, float)) or
              not math.isfinite(self.best_value) or type(self.best_epoch) is not int or
              not 0 <= self.best_epoch <= self.last_epoch or self.bad_epochs > self.last_epoch - self.best_epoch):
            raise ValueError("Invalid best metric/epoch state")

    @classmethod
    def from_config(cls, config):
        return cls(config.early_stopping_mode, config.early_stopping_patience,
                   config.early_stopping_min_delta, config.early_stopping_metric)

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience

    def update(self, metric, epoch):
        if not isinstance(metric, ValidationMetric) or metric.name != self.metric_name:
            raise ValueError("Configured validation metric required")
        if type(epoch) is not int or epoch <= self.last_epoch:
            raise ValueError("Epoch must increase monotonically")
        improved = self.best_value is None
        if not improved:
            improved = (metric.value < self.best_value - self.min_delta if self.mode == "min"
                        else metric.value > self.best_value + self.min_delta)
        self.last_epoch = epoch
        if improved:
            self.best_value, self.best_epoch, self.bad_epochs = metric.value, epoch, 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.__dataclass_fields__):
            raise ValueError("Early-stopping schema mismatch")
        candidate = EarlyStopping(**state)
        for name in ("mode", "patience", "min_delta", "metric_name"):
            if getattr(candidate, name) != getattr(self, name):
                raise ValueError("Early-stopping policy mismatch")
        self.__dict__.update(candidate.__dict__)
