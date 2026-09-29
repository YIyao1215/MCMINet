"""Training-only mixed-class batch composition without replacement."""

from collections.abc import Iterator, Sequence
from numbers import Integral

import torch
from torch import Tensor
from torch.utils.data import Sampler

from mcminet.data.patient_dataset import _binary_label


class ClassAwareBatchSampler(Sampler[list[int]]):
    """Intended for training only; never oversamples or computes class weights.

    Validation/test should use ordinary sequential/non-class-aware batching,
    not this sampler for performance optimization. No cohort logic or overlap
    checks are implemented here; indices refer only to the supplied labels.

    Exactly min(n0,n1,N//batch_size) full batches are mixed, the maximum possible
    without replacement. Remaining batches may be single-class; a single-class
    dataset reduces to ordinary batching. Incomplete batches need not be mixed.
    drop_last=False covers every index exactly once; True omits only leftovers.

    shuffle=False preserves original relative order within each class. With
    shuffle=True, a supplied CPU Generator is advanced normally across epochs;
    equal initial generator states reproduce batches. If omitted, a private
    entropy-seeded CPU generator is used. Global RNG state is never changed.
    """

    def __init__(self, labels: Sequence[int] | Tensor, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False,
                 generator: torch.Generator | None = None) -> None:
        if isinstance(labels, Tensor) and labels.ndim != 1:
            raise ValueError("labels must be one-dimensional.")
        self.labels = tuple(_binary_label(label) for label in labels)
        if not self.labels:
            raise ValueError("Training labels must be nonempty.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 2:
            raise ValueError("batch_size must be an integer >= 2.")
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise TypeError("shuffle and drop_last must be bool.")
        if generator is not None and (not isinstance(generator, torch.Generator) or generator.device.type != 'cpu'):
            raise ValueError("generator must be a CPU torch.Generator.")
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator if generator is not None else torch.Generator()
        if generator is None:
            self.generator.seed()

    def __len__(self) -> int:
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        queues = [[i for i, label in enumerate(self.labels) if label == c] for c in (0, 1)]
        def permute(values):
            if self.shuffle:
                return [values[i] for i in torch.randperm(len(values), generator=self.generator).tolist()]
            return values
        queues = [permute(q) for q in queues]
        full = len(self.labels) // self.batch_size
        mixed = min(len(queues[0]), len(queues[1]), full)
        # Reserve one slot per class for every feasible mixed full batch before
        # filling any extras, so early batches cannot exhaust the minority.
        remaining = sorted(queues[0][mixed:] + queues[1][mixed:])
        remaining = permute(remaining)
        tokens = [self.labels[i] for i in remaining]
        cursor = 0
        positions = [0, 0]
        for b in range(len(self)):
            size = min(self.batch_size, len(self.labels) - b * self.batch_size)
            plan = [0, 1] if b < mixed else []
            needed = size - len(plan)
            plan.extend(tokens[cursor:cursor + needed])
            cursor += needed
            plan = permute(plan)
            batch = []
            for label in plan:
                batch.append(queues[label][positions[label]])
                positions[label] += 1
            yield batch
