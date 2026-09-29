"""Reusable cached-path epoch mechanics. Optimizer execution is caller-controlled."""
from dataclasses import dataclass
import math
import torch
from mcminet.data.cached_patient_dataset import CachedPatientSample, MRI_FIELDS
from mcminet.models.cached_batched_mcminet import CachedBatchedMCMINet
from mcminet.training.optimizer_builder import audit_optimizer
from mcminet.training.training_config import validate_objective
from mcminet.training.training_policy import audit_training_policy
from mcminet.training.scheduler import ValidationMetric

LOSS_NAMES = ("total", "classification", "intra_modal", "cross_modal", "hybrid_proxy", "specific_proxy")


@dataclass(frozen=True)
class EpochResult:
    source: str
    patient_count: int
    batch_count: int
    losses: dict

    def __post_init__(self):
        if self.source not in ("training", "validation"):
            raise ValueError("Only training/internal validation epoch results supported")
        if type(self.patient_count) is not int or type(self.batch_count) is not int or not 0 < self.batch_count <= self.patient_count:
            raise ValueError("Empty/invalid epoch")
        if set(self.losses) != set(LOSS_NAMES) or any(not math.isfinite(v) for v in self.losses.values()):
            raise ValueError("Invalid epoch losses")

    def selection(self, name="total"):
        if self.source != "validation":
            raise ValueError("Training losses cannot select a model")
        return ValidationMetric(name, self.losses[name])

    def to_dict(self):
        return dict(source=self.source, patient_count=self.patient_count,
                    batch_count=self.batch_count, losses=dict(self.losses))


def _prepare_batch(batch, device):
    required = {"patient_ids", *MRI_FIELDS, "wsi_features_list", "wsi_coordinates_list", "labels"}
    if not isinstance(batch, dict) or set(batch) != required:
        raise ValueError("Expected validated cached-collate batch schema")
    labels = batch["labels"]
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.device.type != "cpu" or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Labels must be nonempty CPU long [B]")
    count = len(labels)
    if len(batch["patient_ids"]) != count or len(set(batch["patient_ids"])) != count:
        raise ValueError("Missing/duplicate patient IDs")
    for name in ("wsi_features_list", "wsi_coordinates_list"):
        if not isinstance(batch[name], (list, tuple)) or len(batch[name]) != count:
            raise ValueError("WSI patient count mismatch")
    for name in MRI_FIELDS:
        roi = batch[name]
        if not isinstance(roi, torch.Tensor) or roi.ndim != 4 or len(roi) != count:
            raise ValueError("MRI batch shape mismatch")
    for i, pid in enumerate(batch["patient_ids"]):
        CachedPatientSample(pid, *(batch[name][i] for name in MRI_FIELDS),
                            batch["wsi_features_list"][i], batch["wsi_coordinates_list"][i], int(labels[i]))
    # Cached features and coordinates deliberately remain CPU. The validated cached
    # branch handles transfer of its detached concatenated feature tensor to GAT.
    return ([batch[name].to(device) for name in MRI_FIELDS],
            batch["wsi_features_list"], batch["wsi_coordinates_list"], labels.to(device))


def _run(model, objective, batches, offline_encoder, config, device, optimizer):
    if not isinstance(model, CachedBatchedMCMINet):
        raise ValueError("CachedBatchedMCMINet required")
    validate_objective(objective, config)
    audit_training_policy(model, objective, offline_encoder)
    device = torch.device(device)
    for module in (model, objective):
        if any(t.device != device for t in list(module.parameters()) + list(module.buffers())):
            raise ValueError("Move model/objective to the explicit device before building optimizer")
    training = optimizer is not None
    if training:
        audit_optimizer(optimizer, model, objective, offline_encoder, config)
    model.train(training)
    objective.train(training)
    totals = {name: 0.0 for name in LOSS_NAMES}
    patients = batch_count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in batches:
            rois, features, coordinates, labels = _prepare_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(*rois, features, coordinates, return_embeddings=True)
            losses = objective(output["logits"], output["z_mri"], output["z_wsi"], labels)
            if set(losses) != set(LOSS_NAMES) or any(v.ndim != 0 or not torch.isfinite(v) for v in losses.values()):
                raise ValueError("Nonfinite/malformed objective; optimizer step not executed")
            if training:
                losses["total"].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for group in optimizer.param_groups for p in group["params"]):
                    optimizer.zero_grad(set_to_none=True)
                    raise ValueError("Nonfinite gradient; optimizer step not executed")
                optimizer.step()
            count = len(labels)
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * count
            patients += count
            batch_count += 1
    if patients == 0:
        raise ValueError("Empty epoch")
    return EpochResult("training" if training else "validation", patients, batch_count,
                       {name: value / patients for name, value in totals.items()})


def train_one_epoch(model, objective, batches, optimizer, offline_encoder, config, *, device="cpu"):
    if optimizer is None:
        raise ValueError("Training requires an audited optimizer")
    return _run(model, objective, batches, offline_encoder, config, device, optimizer)


def evaluate_one_epoch(model, objective, batches, offline_encoder, config, *, device="cpu"):
    return _run(model, objective, batches, offline_encoder, config, device, None)
