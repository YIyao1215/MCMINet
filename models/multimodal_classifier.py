"""Patient-level multimodal fusion classifier for MRI and WSI embeddings."""
from mcminet.config import default


from math import isfinite
from numbers import Real

import torch
from torch import Tensor, nn


class MultimodalClassifier(nn.Module):
    """Consume embeddings as supplied, concatenate [MRI; WSI], return logits.

    Default: [B, 256] + [B, 256] -> Linear(512, 256) -> ReLU ->
    Dropout(0.2) -> Linear(256, 1) -> [B]. No normalization or sigmoid.
    Inputs and model parameters must share device and floating dtype.
    """

    def __init__(self, embedding_dim: int = default("model.embedding_dim"), hidden_dim: int = default("model.classifier_hidden_dim"), dropout: float = default("model.classifier_dropout")) -> None:
        super().__init__()
        for name, value in (("embedding_dim", embedding_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (isinstance(dropout, bool) or not isinstance(dropout, Real)
                or not isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z_mri: Tensor, z_wsi: Tensor) -> Tensor:
        """Map two finite floating tensors [B, D] to raw binary logits [B]."""
        for name, value in (("z_mri", z_mri), ("z_wsi", z_wsi)):
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != self.embedding_dim:
                raise ValueError(f"{name} must have shape [B, {self.embedding_dim}].")
            if value.shape[0] == 0:
                raise ValueError("Batch must be nonempty.")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating-point.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must not contain NaN or Inf.")
            if value.device != self.fc1.weight.device or value.dtype != self.fc1.weight.dtype:
                raise ValueError("Embeddings and classifier must share device and dtype.")
        if z_mri.shape[0] != z_wsi.shape[0]:
            raise ValueError("MRI and WSI batch sizes must match.")
        hidden = self.dropout(self.activation(self.fc1(torch.cat((z_mri, z_wsi), dim=1))))
        return self.fc2(hidden).squeeze(-1)
