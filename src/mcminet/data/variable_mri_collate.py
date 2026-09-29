"""Native-size MRI lists with the existing cached WSI/label sample semantics."""
import torch
from mcminet.data.cached_patient_dataset import MRI_FIELDS, validate_sample


def variable_mri_collate(samples):
    if not samples:
        raise ValueError("Empty patient batch")
    for sample in samples:
        validate_sample(sample)
    ids = [s.patient_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient in batch")
    # Preserve exact objects, dtype, units, ordering and native spatial extents.
    return dict(patient_ids=ids,
                **{name: [getattr(s, name) for s in samples] for name in MRI_FIELDS},
                wsi_features_list=[s.wsi_features for s in samples],
                wsi_coordinates_list=[s.wsi_coordinates for s in samples],
                labels=torch.tensor([s.label for s in samples], dtype=torch.long, device="cpu"))
