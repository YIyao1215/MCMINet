"""Explicit orchestration-level cohort contracts, without dataset edits."""
from mcminet.config import default
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CohortLoader:
    loader: object
    role: str
    patient_labels: tuple
    batch_size: int = default("training.batch_size")
    synthetic: bool = False

    def validate(self, expected_role):
        if self.role != expected_role or self.role not in ('TRAIN', 'INTERNAL_VALIDATION'):
            raise ValueError('EXTERNAL_TEST/incorrect cohort role forbidden')
        if type(self.batch_size) is not int or self.batch_size != default("training.batch_size"):
            raise ValueError('batch_size differs from the canonical candidate')
        for owner in (self.loader, getattr(self.loader, 'dataset', None),
                      getattr(self.loader, 'batch_sampler', None)):
            if owner is None:
                continue
            for attr in ('role', 'cohort_role'):
                if hasattr(owner, attr) and getattr(owner, attr) != expected_role:
                    raise ValueError('Loader/dataset cohort role conflicts with explicit contract')
            if getattr(owner, 'batch_size', None) not in (None, self.batch_size):
                raise ValueError('Loader batch_size differs from candidate contract')
            if getattr(owner, 'drop_last', False):
                raise ValueError('Complete patient epochs require drop_last=False')
        if not isinstance(self.patient_labels, tuple) or not self.patient_labels:
            raise ValueError('Nonempty immutable patient roster required')
        ids = []
        for row in self.patient_labels:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError('Roster rows must be (patient_id, binary_label) tuples')
            pid, label = row
            if not isinstance(pid, str) or not pid or type(label) is not int or label not in (0, 1):
                raise ValueError('Invalid cohort patient/label')
            ids.append(pid)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate cohort patient')
        if expected_role == 'INTERNAL_VALIDATION' and {r[1] for r in self.patient_labels} != {0, 1}:
            raise ValueError('Validation AUROC requires both classes')

    def batches(self):
        expected = dict(self.patient_labels)
        seen = set()
        for batch in self.loader:
            ids = batch['patient_ids']
            labels = batch['labels']
            if not isinstance(ids, list) or not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.ndim != 1:
                raise ValueError('Native patient IDs and long labels required')
            remaining = len(expected) - len(seen)
            if len(ids) != min(self.batch_size, remaining) or len(labels) != len(ids):
                raise ValueError('Native batch must use candidate batch_size with only final remainder allowed')
            for pid, label in zip(ids, labels.tolist()):
                if pid in seen or pid not in expected or label != expected[pid]:
                    raise ValueError('Duplicate/foreign/mislabeled cohort patient')
                seen.add(pid)
            yield batch
        if seen != set(expected):
            raise ValueError('Incomplete cohort epoch')


def validate_cohorts(train, validation):
    train.validate('TRAIN')
    validation.validate('INTERNAL_VALIDATION')
    if {r[0] for r in train.patient_labels} & {r[0] for r in validation.patient_labels}:
        raise ValueError('Train/internal-validation patient overlap')
