"""Explicit cache reads and prepared MRI collation; no regeneration or graph work."""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset
from mcminet.data.patient_dataset import _binary_label
from mcminet.data.wsi_feature_cache import load_cache,validate_tensors

MRI_FIELDS=('tumor_roi','peritumoral_roi','lymph_node_roi')


@dataclass(frozen=True)
class CachedPatientSample:
    patient_id: str
    tumor_roi: torch.Tensor
    peritumoral_roi: torch.Tensor
    lymph_node_roi: torch.Tensor
    wsi_features: torch.Tensor
    wsi_coordinates: torch.Tensor
    label: int

    def __post_init__(self):validate_sample(self)


def validate_sample(sample):
    if not isinstance(sample,CachedPatientSample):raise TypeError('CachedPatientSample required')
    if not isinstance(sample.patient_id,str) or not sample.patient_id.strip():raise ValueError('Patient ID required')
    _binary_label(sample.label)
    for name in MRI_FIELDS:
        v=getattr(sample,name)
        if not isinstance(v,torch.Tensor) or v.ndim!=3 or v.shape[0]!=1 or not all(v.shape):raise ValueError('MRI must be nonempty [1,H,W]')
        if v.dtype!=torch.float32 or v.device.type!='cpu' or not torch.isfinite(v).all():raise ValueError('MRI must be finite CPU float32')
    validate_tensors(sample.wsi_features,sample.wsi_coordinates)


@dataclass(frozen=True)
class CachedPatientRecord:
    patient_id: str
    tumor_roi: torch.Tensor
    peritumoral_roi: torch.Tensor
    lymph_node_roi: torch.Tensor
    cache_path: Path
    expected_provenance: dict
    label: int


class CachedPatientDataset(Dataset):
    """Lazy explicit cache consumption. Missing/stale cache fails, never regenerates.

    Records are a caller-selected cohort; no split inference or cross-cohort mixing.
    Use validated ClassAwareBatchSampler(dataset.labels, ...) for training only.
    """
    def __init__(self,records):
        self.records=tuple(records);seen=set()
        for r in self.records:
            if not isinstance(r,CachedPatientRecord) or not r.patient_id or r.patient_id in seen:raise ValueError('Invalid/duplicate cached record')
            if r.expected_provenance['patient_id']!=r.patient_id:raise ValueError('Cache patient identity mismatch')
            _binary_label(r.label);seen.add(r.patient_id)
        self.labels=tuple(_binary_label(r.label) for r in self.records)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];cache=load_cache(r.cache_path,r.expected_provenance)
        return CachedPatientSample(r.patient_id,r.tumor_roi,r.peritumoral_roi,r.lymph_node_roi,
                                   cache['features'],cache['coordinates'],r.label)


def cached_collate_fn(samples):
    if not samples:raise ValueError('Empty batch')
    for s in samples:validate_sample(s)
    if len({s.patient_id for s in samples})!=len(samples):raise ValueError('Duplicate patient in batch')
    for name in MRI_FIELDS:
        first=getattr(samples[0],name)
        if any(getattr(s,name).shape!=first.shape for s in samples):
            raise ValueError(name+' must be stackable within ROI type; no implicit resize/padding')
    return dict(patient_ids=[s.patient_id for s in samples],
        **{name:torch.stack([getattr(s,name) for s in samples]) for name in MRI_FIELDS},
        wsi_features_list=[s.wsi_features for s in samples],
        wsi_coordinates_list=[s.wsi_coordinates for s in samples],
        labels=torch.tensor([_binary_label(s.label) for s in samples],dtype=torch.long,device='cpu'))
