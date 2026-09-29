"""Immutable training adapter derived from the canonical candidate YAML."""
from mcminet.config import default, load_config
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real

DEFAULT_NOTICE = "PRESPECIFIED FORMAL CANDIDATE — INTERNAL VALIDATION REQUIRED"


def finite_number(name, value, *, minimum=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} is out of range")


@dataclass(frozen=True)
class TrainingConfig:
    """All numeric choices are configurable and provisional, including temperatures."""
    batch_size: int = default("training.batch_size")
    scheduler_metric: str = default("scheduler.monitor")
    scheduler_mode: str = default("scheduler.mode")
    scheduler_threshold_mode: str = default("scheduler.threshold_mode")
    early_stopping_metric: str = default("early_stopping.monitor")
    early_stopping_mode: str = default("early_stopping.mode")
    bias_weight_decay: float = default("optimizer.bias_weight_decay")
    batchnorm_weight_decay: float = default("optimizer.batchnorm_weight_decay")
    proxy_weight_decay: float = default("optimizer.proxy_weight_decay")
    seed: int = default("reproducibility.primary_seed")
    optimizer_type: str = default("optimizer.name")
    mri_backbone_lr: float = default("optimizer.lr_mri_pretrained")
    mri_new_lr: float = default("optimizer.lr_mri_new")
    wsi_lr: float = default("optimizer.lr_wsi")
    classifier_lr: float = default("optimizer.lr_classifier")
    proxy_lr: float = default("optimizer.lr_proxy")
    weight_decay: float = default("optimizer.weight_decay")
    adam_beta1: float = default("optimizer.betas")[0]
    adam_beta2: float = default("optimizer.betas")[1]
    adam_eps: float = default("optimizer.eps")
    lambda_cls: float = default("objective.lambda_cls")
    lambda_intra: float = default("objective.lambda_intra")
    lambda_cross: float = default("objective.lambda_cross")
    lambda_hybrid: float = default("objective.lambda_hybrid")
    lambda_specific: float = default("objective.lambda_specific")
    tau_intra: float = default("objective.tau_intra")
    tau_cross: float = default("objective.tau_cross")
    tau_hybrid: float = default("objective.tau_hybrid")
    tau_specific: float = default("objective.tau_specific")
    max_epochs: int = default("training.max_epochs")
    scheduler_type: str = default("scheduler.name")
    scheduler_factor: float = default("scheduler.factor")
    scheduler_patience: int = default("scheduler.patience")
    scheduler_threshold: float = default("scheduler.threshold")
    scheduler_cooldown: int = default("scheduler.cooldown")
    scheduler_min_lr: float = default("scheduler.min_lr")
    scheduler_eps: float = default("scheduler.eps")
    selection_metric: str = default("evaluation.selection_metric")
    selection_mode: str = default("evaluation.selection_mode")
    early_stopping_patience: int = default("early_stopping.patience")
    early_stopping_min_delta: float = default("early_stopping.min_delta")
    deterministic_algorithms: bool = default("reproducibility.deterministic_algorithms")
    checkpoint_directory: str = "artifacts/future_training/checkpoints"
    history_filename: str = "history.json"

    def __post_init__(self):
        integer_fields = {
            "batch_size": (1, None), "seed": (0, 2**32-1), "max_epochs": (1, None),
            "scheduler_patience": (0, None), "scheduler_cooldown": (0, None),
            "early_stopping_patience": (1, None),
        }
        for name, (low, high) in integer_fields.items():
            value = getattr(self, name)
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f"Invalid {name}")
        positive = ("mri_backbone_lr", "mri_new_lr", "wsi_lr", "classifier_lr",
                    "proxy_lr", "adam_eps", "tau_intra", "tau_cross", "tau_hybrid", "tau_specific")
        nonnegative = ("bias_weight_decay", "batchnorm_weight_decay", "proxy_weight_decay", "weight_decay", "lambda_cls", "lambda_intra", "lambda_cross",
                       "lambda_hybrid", "lambda_specific", "scheduler_threshold",
                       "scheduler_min_lr", "scheduler_eps", "early_stopping_min_delta")
        for name in positive:
            finite_number(name, getattr(self, name), positive=True)
        for name in nonnegative:
            finite_number(name, getattr(self, name))
        for name in ("adam_beta1", "adam_beta2", "scheduler_factor"):
            finite_number(name, getattr(self, name))
            if getattr(self, name) >= 1 or (name == "scheduler_factor" and getattr(self, name) == 0):
                raise ValueError(f"Invalid {name}")
        if self.scheduler_min_lr > min(getattr(self, name) for name in positive[:5]):
            raise ValueError("scheduler_min_lr cannot exceed an initial group LR")
        if self.optimizer_type != "adamw":
            raise ValueError("Only the AdamW implementation is currently supported; no fallback")
        if self.scheduler_type not in ("plateau", "none"):
            raise ValueError("Unknown scheduler type")
        if self.scheduler_metric != default("scheduler.monitor") or self.scheduler_mode != default("scheduler.mode") or self.scheduler_threshold_mode != default("scheduler.threshold_mode"):
            raise ValueError("Scheduler must monitor internal-validation total loss")
        if self.early_stopping_metric != default("early_stopping.monitor") or self.early_stopping_mode != default("early_stopping.mode"):
            raise ValueError("Early stopping requires internal-validation AUROC")
        if any(getattr(self, n) != default("optimizer." + n) for n in ("bias_weight_decay", "batchnorm_weight_decay", "proxy_weight_decay")):
            raise ValueError("Bias, BatchNorm and proxy weight decay must be zero")
        if self.selection_metric != default("evaluation.selection_metric") or self.selection_mode != default("evaluation.selection_mode"):
            raise ValueError("Selection requires internal-validation AUROC")
        if self.selection_mode not in ("min", "max"):
            raise ValueError("Selection mode must be min/max")
        if not isinstance(self.selection_metric, str) or not self.selection_metric.strip():
            raise ValueError("A validation metric name is required")
        if type(self.deterministic_algorithms) is not bool:
            raise ValueError("deterministic_algorithms must be bool")
        for name in ("checkpoint_directory", "history_filename"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Invalid {name}")

    @classmethod
    def from_yaml(cls, path=None, *, seed=None):
        return cls.from_config(load_config(path), seed=seed)

    @classmethod
    def from_config(cls, config, *, seed=None):
        from mcminet.config import validate_config
        validate_config(config)
        def get(key):
            value = config
            for part in key.split('.'):
                value = value[part]
            return value
        values = {name: get(key) for name, key in CONFIG_FIELDS.items()}
        values.update(adam_beta1=config['optimizer']['betas'][0], adam_beta2=config['optimizer']['betas'][1])
        if seed is not None:
            values['seed'] = seed
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Configuration schema mismatch")
        return cls(**value)


def objective_settings(config):
    names = ("lambda_cls", "lambda_intra", "lambda_cross", "lambda_hybrid",
             "lambda_specific", "tau_intra", "tau_cross", "tau_hybrid", "tau_specific")
    return {name: getattr(config, name) for name in names}


def build_objective(config):
    from mcminet.losses.mcminet_objective import MCMINetObjective
    return MCMINetObjective(**objective_settings(config))


def validate_objective(objective, config):
    from mcminet.losses.mcminet_objective import MCMINetObjective
    if not isinstance(objective, MCMINetObjective):
        raise ValueError("The validated MCMINetObjective is required")
    for name, value in objective_settings(config).items():
        owner = objective.proxy_metric if name.startswith("tau_") else objective
        if getattr(owner, name) != value:
            raise ValueError(f"Objective/config mismatch: {name}")


CONFIG_FIELDS = {'seed': 'reproducibility.primary_seed',
 'optimizer_type': 'optimizer.name',
 'mri_backbone_lr': 'optimizer.lr_mri_pretrained',
 'mri_new_lr': 'optimizer.lr_mri_new',
 'wsi_lr': 'optimizer.lr_wsi',
 'classifier_lr': 'optimizer.lr_classifier',
 'proxy_lr': 'optimizer.lr_proxy',
 'weight_decay': 'optimizer.weight_decay',
 'adam_eps': 'optimizer.eps',
 'max_epochs': 'training.max_epochs',
 'scheduler_type': 'scheduler.name',
 'selection_metric': 'evaluation.selection_metric',
 'selection_mode': 'evaluation.selection_mode',
 'early_stopping_patience': 'early_stopping.patience',
 'early_stopping_min_delta': 'early_stopping.min_delta',
 'deterministic_algorithms': 'reproducibility.deterministic_algorithms',
 'lambda_cls': 'objective.lambda_cls',
 'lambda_intra': 'objective.lambda_intra',
 'lambda_cross': 'objective.lambda_cross',
 'lambda_hybrid': 'objective.lambda_hybrid',
 'lambda_specific': 'objective.lambda_specific',
 'tau_intra': 'objective.tau_intra',
 'tau_cross': 'objective.tau_cross',
 'tau_hybrid': 'objective.tau_hybrid',
 'tau_specific': 'objective.tau_specific',
 'scheduler_factor': 'scheduler.factor',
 'scheduler_patience': 'scheduler.patience',
 'scheduler_threshold': 'scheduler.threshold',
 'scheduler_cooldown': 'scheduler.cooldown',
 'scheduler_min_lr': 'scheduler.min_lr',
 'scheduler_eps': 'scheduler.eps',
 'batch_size': 'training.batch_size',
 'scheduler_metric': 'scheduler.monitor',
 'scheduler_mode': 'scheduler.mode',
 'scheduler_threshold_mode': 'scheduler.threshold_mode',
 'early_stopping_metric': 'early_stopping.monitor',
 'early_stopping_mode': 'early_stopping.mode',
 'bias_weight_decay': 'optimizer.bias_weight_decay',
 'batchnorm_weight_decay': 'optimizer.batchnorm_weight_decay',
 'proxy_weight_decay': 'optimizer.proxy_weight_decay'}
