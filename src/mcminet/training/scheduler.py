"""Selection quantities carry explicit validation provenance."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ValidationMetric:
    name: str
    value: float
    source: str = "validation"

    def __post_init__(self):
        if self.source != "validation":
            raise ValueError("Only internal validation metrics may select a model")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Metric name required")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError("Nonfinite/invalid validation metric; selection not advanced")


class ValidationScheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.scheduler = None
        if config.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=config.scheduler_mode, factor=config.scheduler_factor,
                patience=config.scheduler_patience, threshold=config.scheduler_threshold,
                threshold_mode=config.scheduler_threshold_mode, cooldown=config.scheduler_cooldown,
                min_lr=config.scheduler_min_lr, eps=config.scheduler_eps,
            )

    def step(self, metric):
        if not isinstance(metric, ValidationMetric) or metric.name != self.config.scheduler_metric:
            raise ValueError("Configured validation metric required")
        if self.scheduler is not None:
            self.scheduler.step(metric.value)

    def state_dict(self):
        return dict(type=self.config.scheduler_type, metric=self.config.scheduler_metric,
                    mode=self.config.scheduler_mode,
                    state=None if self.scheduler is None else self.scheduler.state_dict())

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"type", "metric", "mode", "state"}:
            raise ValueError("Invalid scheduler schema")
        for key, expected in [("type", self.config.scheduler_type), ("metric", self.config.scheduler_metric), ("mode", self.config.scheduler_mode)]:
            if state[key] != expected:
                raise ValueError("Scheduler configuration mismatch")
        if self.scheduler is None:
            if state["state"] is not None:
                raise ValueError("Unexpected scheduler state")
        else:
            if not isinstance(state["state"], dict) or set(state["state"]) != set(self.scheduler.state_dict()):
                raise ValueError("Scheduler state structure mismatch")
            values = state["state"]
            current = self.scheduler.state_dict()
            dynamic = {"last_epoch", "_last_lr", "best", "cooldown_counter", "num_bad_epochs"}
            for key in set(current) - dynamic:
                if values[key] != current[key]:
                    raise ValueError("Scheduler policy state differs: " + key)
            for key in ("last_epoch", "cooldown_counter", "num_bad_epochs"):
                if type(values[key]) is not int or values[key] < 0:
                    raise ValueError("Invalid scheduler counter")
            if not isinstance(values["best"], (int, float)) or math.isnan(values["best"]):
                raise ValueError("Invalid scheduler best value")
            lrs = values["_last_lr"]
            if not isinstance(lrs, list) or len(lrs) != len(self.optimizer.param_groups) or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in lrs):
                raise ValueError("Invalid scheduler learning rates")
            self.scheduler.load_state_dict(values)
