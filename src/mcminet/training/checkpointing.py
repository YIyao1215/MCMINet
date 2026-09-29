"""Strict, atomic, trusted-local epoch-boundary checkpoints for the cached model."""
import copy
import re
from pathlib import Path
import torch
from mcminet.data.wsi_feature_cache import sha256_file, preprocessing_identity
from mcminet.utils.state_fingerprint import state_fingerprint
from mcminet.training.training_config import TrainingConfig
from mcminet.training.optimizer_builder import audit_optimizer, initial_group_lr
from mcminet.config import load_config, scientific_settings
from mcminet.training.scheduler import ValidationScheduler
from mcminet.training.early_stopping import EarlyStopping
from mcminet.training.training_history import TrainingHistory, atomic_write
from mcminet.training.reproducibility import capture_rng, restore_rng, run_metadata

SCHEMA_VERSION = 2
ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = (
    "config.py", "training/optimizer_builder.py", "training/model_contract.py",
    "training/scientific_run_history.py", "data/mri_preprocessing.py",
    "training/training_policy.py", "data/wsi_feature_cache.py",
    "models/cached_wsi_branch.py", "models/cached_batched_mcminet.py",
    "data/cached_patient_dataset.py", "losses/mcminet_objective.py",
    "losses/proxy_metric_learning.py", "models/mri_branch.py",
    "models/mri_roi_encoder.py", "models/wsi_gat_encoder.py",
    "models/multimodal_classifier.py", "data/wsi_graph_builder.py",
)


def training_identity(cache_identity, model=None, config=None):
    """Caller supplies the three hashes from trusted current release expectations.

    Patient-specific WSI/GeoJSON identity still belongs to each cache record and
    remains validated by CachedPatientDataset. No labels/splits enter this identity.
    """
    keys = {"encoder_state_sha256", "checkpoint_sha256", "preprocessing_sha256"}
    if not isinstance(cache_identity, dict) or set(cache_identity) != keys:
        raise ValueError("Exact fixed-encoder/preprocessing identity required")
    if any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in cache_identity.values()):
        raise ValueError("Cache identity must contain SHA-256 values")
    if cache_identity["preprocessing_sha256"] != preprocessing_identity()["sha256"]:
        raise ValueError("Cache preprocessing identity differs from the current candidate")
    from mcminet.training.model_contract import validate_model_contract
    settings = getattr(model, 'candidate_config', None) or load_config()
    if model is not None:
        validate_model_contract(model, settings, synthetic=getattr(model, 'smoke_test', False))
    graph = dict(model.wsi_branch.graph_config) if model is not None else settings['graph']
    return dict(encoder_initialization=getattr(model, 'encoder_initialization', None),
                smoke_test=getattr(model, 'smoke_test', False),
                candidate_configuration=scientific_settings(settings),
                training_configuration=None if config is None else config.to_dict(),
                preprocessing_version=settings['WSI']['preprocessing']['version'],
                cache=copy.deepcopy(cache_identity),
                policy_source_sha256={p: sha256_file(ROOT / p) for p in POLICY_FILES},
                graph=graph,
                feature_dimension=settings["WSI"]["node_feature_dim"], feature_dtype="torch.float32")


def _tensor_state(saved, current, name):
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise ValueError(f"{name} state keys differ; partial loading forbidden")
    for key, value in saved.items():
        expected = current[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"{name} tensor shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {name} tensor: {key}")


def _optimizer_state(saved, optimizer):
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer state schema")
    current = optimizer.state_dict()
    if len(saved["param_groups"]) != len(current["param_groups"]):
        raise ValueError("Optimizer group count mismatch")
    id_to_param = {}
    for loaded, expected, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        if set(loaded) != set(expected):
            raise ValueError("Optimizer group schema differs")
        for key in expected:
            if key != "lr" and loaded[key] != expected[key]:
                raise ValueError(f"Optimizer group/order/config differs: {key}")
        lr = loaded["lr"]
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not torch.isfinite(torch.tensor(lr)) or lr < 0:
            raise ValueError("Invalid restored learning rate")
        id_to_param.update(zip(loaded["params"], live["params"]))
    if not isinstance(saved["state"], dict) or not set(saved["state"]) <= set(id_to_param):
        raise ValueError("Foreign optimizer state")
    for identifier, state in saved["state"].items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW moment schema differs")
        p = id_to_param[identifier]
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid AdamW moments")
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 0:
            raise ValueError("Invalid AdamW step")


def _selection_state(epoch, early_state, history_state, scheduler_state, optimizer, config):
    if type(epoch) is not int or not 0 <= epoch < config.max_epochs:
        raise ValueError("Epoch outside configured range")
    early = EarlyStopping.from_config(config)
    early.load_state_dict(early_state)
    history = TrainingHistory()
    history.load_state_dict(history_state)
    rows = history.state_dict()
    if [r["epoch"] for r in rows] != list(range(epoch + 1)) or early.last_epoch != epoch:
        raise ValueError("Only complete epoch-boundary histories can resume")
    replay = EarlyStopping.from_config(config)
    # A shallow optimizer view permits scheduler replay without touching live
    # group LRs or any model parameters; no optimizer is constructed or stepped.
    view = copy.copy(optimizer)
    view.param_groups = [dict(group, lr=initial_group_lr(config, group))
                         for group in optimizer.param_groups]
    replay_scheduler = ValidationScheduler(view, config)
    from mcminet.training.scheduler import ValidationMetric
    for row in rows:
        metric = ValidationMetric(**row["selection"])
        replay.update(metric, row["epoch"])
        replay_scheduler.step(ValidationMetric(config.scheduler_metric, row["validation"]["losses"][config.scheduler_metric]))
        if row["learning_rates"] != {g["name"]:g["lr"] for g in view.param_groups}:
            raise ValueError("History learning rates disagree with scheduler replay")
    if replay.state_dict() != early_state:
        raise ValueError("Best metric/early stopping disagrees with validation history")
    scheduler = ValidationScheduler(optimizer, config)
    scheduler.load_state_dict(scheduler_state)
    if replay_scheduler.state_dict() != scheduler_state:
        raise ValueError("Scheduler state disagrees with validation history")
    return early, history


def _modes(module):
    return {name: child.training for name, child in module.named_modules()}


def _restore_modes(module, modes):
    if not isinstance(modes, dict) or set(modes) != set(_modes(module)) or any(type(v) is not bool for v in modes.values()):
        raise ValueError("Module mode schema mismatch")
    for name, child in module.named_modules():
        child.training = modes[name]


def save_checkpoint(path, *, model, objective, optimizer, scheduler, early_stopping,
                    history, config, epoch, offline_encoder, cache_identity,
                    device="cpu", generator=None):
    audit_optimizer(optimizer, model, objective, offline_encoder, config)
    identity = training_identity(cache_identity, model, config)
    if state_fingerprint(offline_encoder) != cache_identity["encoder_state_sha256"]:
        raise ValueError("Actual offline encoder differs from expected cache identity")
    if scheduler.optimizer is not optimizer or scheduler.config != config:
        raise ValueError("Scheduler ownership/config mismatch")
    if any(t.device != torch.device(device) for module in (model, objective)
           for t in list(module.parameters()) + list(module.buffers())):
        raise ValueError("Checkpoint device metadata disagrees with model")
    _selection_state(epoch, early_stopping.state_dict(), history.state_dict(),
                     scheduler.state_dict(), optimizer, config)
    _tensor_state(model.state_dict(), model.state_dict(), "model")
    _tensor_state(objective.state_dict(), objective.state_dict(), "objective")
    _optimizer_state(optimizer.state_dict(), optimizer)
    expected_lrs = (scheduler.state_dict()["state"]["_last_lr"] if config.scheduler_type == "plateau"
                    else [initial_group_lr(config, g) for g in optimizer.param_groups])
    if [g["lr"] for g in optimizer.param_groups] != expected_lrs:
        raise ValueError("Optimizer and scheduler LR state disagree")
    payload = dict(
        schema_version=SCHEMA_VERSION, epoch=epoch, best_metric=early_stopping.best_value,
        model=model.state_dict(), objective=objective.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        early_stopping=early_stopping.state_dict(), history=history.state_dict(),
        config=config.to_dict(), metadata=run_metadata(config, device),
        identity=identity, rng=capture_rng(generator),
        modes=dict(model=_modes(model), objective=_modes(objective)),
        architecture=dict(model=repr(model), objective=repr(objective)),
    )
    atomic_write(path, lambda temporary: torch.save(payload, temporary))


def load_checkpoint(path, *, model, objective, optimizer, scheduler, early_stopping,
                    history, config, offline_encoder, cache_identity,
                    device="cpu", generator=None):
    """Strict same-config/software/device resume; no silent migration or partial load.

    The load is transactional for caller objects/RNG on validation/load failure.
    Gradients are not checkpointed: resume is at an epoch boundary, with the next
    training batch zeroing gradients. Offline encoder is never serialized here.
    """
    audit_optimizer(optimizer, model, objective, offline_encoder, config)
    identity = training_identity(cache_identity, model, config)
    if state_fingerprint(offline_encoder) != cache_identity["encoder_state_sha256"]:
        raise ValueError("Actual offline encoder differs from expected cache identity")
    if scheduler.optimizer is not optimizer or scheduler.config != config:
        raise ValueError("Scheduler ownership/config mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"schema_version", "epoch", "best_metric", "model", "objective", "optimizer",
            "scheduler", "early_stopping", "history", "config", "metadata", "identity", "rng", "modes", "architecture"}
    if not isinstance(saved, dict) or set(saved) != keys or saved["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Incompatible checkpoint schema: candidate v2 requires new decay groups, temperatures and separate AUROC stopping; legacy checkpoints cannot resume")
    if TrainingConfig.from_dict(saved["config"]) != config:
        raise ValueError("Resume requires exact configuration compatibility")
    if saved["identity"] != identity:
        raise ValueError("Validated policy/cache identity mismatch")
    if saved["metadata"] != run_metadata(config, device):
        raise ValueError("Software/device/seed/determinism metadata mismatch")
    if any(t.device != torch.device(device) for module in (model, objective)
           for t in list(module.parameters()) + list(module.buffers())):
        raise ValueError("Resume target must already be on the configured device")
    if saved["architecture"] != dict(model=repr(model), objective=repr(objective)):
        raise ValueError("Model architecture/non-tensor configuration mismatch")
    _tensor_state(saved["model"], model.state_dict(), "model")
    _tensor_state(saved["objective"], objective.state_dict(), "objective")
    _optimizer_state(saved["optimizer"], optimizer)
    _selection_state(saved["epoch"], saved["early_stopping"], saved["history"],
                     saved["scheduler"], optimizer, config)
    expected_lrs = (saved["scheduler"]["state"]["_last_lr"] if config.scheduler_type == "plateau"
                    else [initial_group_lr(config, g) for g in optimizer.param_groups])
    if [g["lr"] for g in saved["optimizer"]["param_groups"]] != expected_lrs:
        raise ValueError("Optimizer and scheduler LR state disagree")
    if saved["best_metric"] != saved["early_stopping"]["best_value"]:
        raise ValueError("Best metric mismatch")
    if not isinstance(saved["modes"], dict) or set(saved["modes"]) != {"model", "objective"}:
        raise ValueError("Invalid modes schema")
    objects = dict(model=model, objective=objective, optimizer=optimizer, scheduler=scheduler,
                   early_stopping=early_stopping, history=history)
    backup = {name: copy.deepcopy(obj.state_dict()) for name, obj in objects.items()}
    old_modes = dict(model=_modes(model), objective=_modes(objective))
    old_rng = capture_rng(generator)
    try:
        for name, obj in objects.items():
            obj.load_state_dict(saved[name])
        _restore_modes(model, saved["modes"]["model"])
        _restore_modes(objective, saved["modes"]["objective"])
        audit_optimizer(optimizer, model, objective, offline_encoder, config)
        restore_rng(saved["rng"], generator)
    except Exception:
        for name, obj in objects.items():
            obj.load_state_dict(backup[name])
        _restore_modes(model, old_modes["model"])
        _restore_modes(objective, old_modes["objective"])
        restore_rng(old_rng, generator)
        raise
    return dict(epoch=saved["epoch"], next_epoch=saved["epoch"] + 1,
                best_metric=saved["best_metric"], config=config, metadata=saved["metadata"])
