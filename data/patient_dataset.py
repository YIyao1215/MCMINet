"""Prepared patient tensors and order-preserving collation; no preprocessing."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _binary_label(value: Any) -> int:
    """Accept integer scalars only: 0=Non-responder, 1=Responder."""
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ValueError("label must be an integer 0 or 1.")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError("label must be an integer 0 or 1.")
    return int(value)


@dataclass(frozen=True)
class PatientSample:
    """One prepared patient; fields are validated but tensor storage is not copied.

    MRI ROIs are [1,H_i,W_i], WSI patches [Ni,3,H,W], coordinates [Ni,2].
    Labels are supplied explicitly: 0=Non-responder, 1=Responder; no MP mapping.
    Callers must not mutate tensor contents while a dataset is in use.
    """

    patient_id: str
    tumor_roi: Tensor
    peritumoral_roi: Tensor
    lymph_node_roi: Tensor
    wsi_patches: Tensor
    wsi_coordinates: Tensor
    label: int

    def __post_init__(self) -> None:
        _validate_sample(self)


def _validate_sample(sample: PatientSample) -> None:
    if not isinstance(sample, PatientSample):
        raise TypeError("samples must contain PatientSample records.")
    if not isinstance(sample.patient_id, str) or not sample.patient_id.strip():
        raise ValueError("patient_id must be a nonempty string.")
    _binary_label(sample.label)
    for name in ('tumor_roi', 'peritumoral_roi', 'lymph_node_roi', 'wsi_patches'):
        value = getattr(sample, name)
        ndim, channels = (4, 3) if name == 'wsi_patches' else (3, 1)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor.")
        if value.ndim != ndim or value.shape[-3] != channels or any(d == 0 for d in value.shape):
            raise ValueError(f"{name} must have nonempty shape " + ('[Ni,3,H,W].' if ndim == 4 else '[1,H,W].'))
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating-point.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
    coords = sample.wsi_coordinates
    if not isinstance(coords, Tensor):
        raise TypeError("wsi_coordinates must be a Tensor.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("wsi_coordinates must have shape [Ni,2].")
    if coords.is_complex() or coords.dtype == torch.bool:
        raise TypeError("wsi_coordinates must contain real numeric values.")
    if not torch.isfinite(coords).all():
        raise ValueError("wsi_coordinates must be finite.")
    if coords.shape[0] != sample.wsi_patches.shape[0]:
        raise ValueError("WSI patch and coordinate counts must match.")


class MCMINetPatientDataset(Dataset[PatientSample]):
    """Wrap supplied records in their original order; an empty dataset is valid.

    Duplicate IDs are rejected within this dataset only. This does not check
    cross-cohort overlap or establish absence of train/validation/test leakage.
    Cohort assignment belongs upstream. Tensor storage is retained unchanged.
    """

    def __init__(self, samples: Sequence[PatientSample]) -> None:
        self._samples = tuple(samples)
        seen = set()
        for sample in self._samples:
            _validate_sample(sample)
            if sample.patient_id in seen:
                raise ValueError(f"Duplicate patient_id: {sample.patient_id!r}.")
            seen.add(sample.patient_id)
        self._labels = tuple(_binary_label(sample.label) for sample in self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PatientSample:
        return self._samples[index]

    @property
    def labels(self) -> tuple[int, ...]:
        return self._labels


def mcminet_collate_fn(samples: Sequence[PatientSample]) -> dict[str, Any]:
    """Stack each MRI ROI type separately, keeping WSI tensors as ordered lists.

    No resize, padding, graph construction, dtype cast or device transfer.
    Same-type MRI shapes/dtypes/devices and WSI patch spatial shapes/dtypes/
    devices must agree within the batch. Coordinates retain their own dtype
    and device. Newly constructed labels are CPU torch.long [B]; device transfer
    of the batch belongs to future training orchestration.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    for sample in samples:
        _validate_sample(sample)
    for name in ('tumor_roi', 'peritumoral_roi', 'lymph_node_roi', 'wsi_patches'):
        first = getattr(samples[0], name)
        for sample in samples[1:]:
            value = getattr(sample, name)
            shape = value.shape[1:] if name == 'wsi_patches' else value.shape
            expected = first.shape[1:] if name == 'wsi_patches' else first.shape
            if shape != expected:
                raise ValueError(f"{name} shapes must match across patients (except WSI Ni).")
            if value.dtype != first.dtype:
                raise ValueError(f"{name} dtype must match across patients.")
            if value.device != first.device:
                raise ValueError(f"{name} device must match across patients.")
    return {
        'patient_ids': [s.patient_id for s in samples],
        **{name: torch.stack([getattr(s, name) for s in samples], dim=0)
           for name in ('tumor_roi', 'peritumoral_roi', 'lymph_node_roi')},
        'wsi_patches_list': [s.wsi_patches for s in samples],
        'wsi_coordinates_list': [s.wsi_coordinates for s in samples],
        'labels': torch.tensor([_binary_label(s.label) for s in samples], dtype=torch.long),
    }
