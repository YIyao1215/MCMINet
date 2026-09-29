"""MRI policy guard around the unchanged release checkpoint schema/loader."""
import torch
from mcminet.models.variable_mri_batching import VariableMRICachedMCMINet, policy_fingerprint
from mcminet.training.mri_batchnorm_policy import audit_mri_batchnorm_policy
from mcminet.training.checkpointing import save_checkpoint, load_checkpoint


def _validate_model(model):
    if not isinstance(model, VariableMRICachedMCMINet):
        raise ValueError("Native MRI model required")
    if model._native_policy_sha256 != policy_fingerprint():
        raise ValueError("Native MRI policy source fingerprint changed")
    return audit_mri_batchnorm_policy(model.mri_branch)


def save_native_checkpoint(path, **kwargs):
    _validate_model(kwargs["model"])
    return save_checkpoint(path, **kwargs)


def load_native_checkpoint(path, **kwargs):
    rows = _validate_model(kwargs["model"])
    # Inspect policy-specific modes before the validated loader can change state.
    # All remaining schema, ownership, identity and transactional checks remain
    # the responsibility of the existing release loader.
    payload = torch.load(path, weights_only=True, map_location="cpu")
    modes = payload.get("modes", {}).get("model", {}) if isinstance(payload, dict) else {}
    if any(modes.get(row["name"]) is not False for row in rows):
        raise ValueError("Checkpoint would enable MRI BN running-stat updates")
    del payload
    return load_checkpoint(path, **kwargs)
