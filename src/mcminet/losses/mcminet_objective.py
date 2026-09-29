"""Composite MCMINet objective: binary classification and proxy metric learning."""
from mcminet.config import default


from math import isfinite
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mcminet.losses.proxy_metric_learning import ProxyMetricLearning


class MCMINetObjective(nn.Module):
    """Combine plain BCEWithLogits with four validated proxy-assisted losses.

    Weights and temperatures come from the formal candidate YAML; they are not
    claimed to be optimized hyperparameters. Tuning belongs to training/internal validation,
    never external test data. Zero weights retain all diagnostic losses.
    This module owns one ProxyMetricLearning, but no classifier or encoders.
    """

    def __init__(
        self, embedding_dim: int = default("model.embedding_dim"), num_classes: int = default("model.num_classes"),
        tau_intra: float = default("objective.tau_intra"), tau_cross: float = default("objective.tau_cross"),
        tau_hybrid: float = default("objective.tau_hybrid"), tau_specific: float = default("objective.tau_specific"),
        lambda_cls: float = default("objective.lambda_cls"), lambda_intra: float = default("objective.lambda_intra"),
        lambda_cross: float = default("objective.lambda_cross"), lambda_hybrid: float = default("objective.lambda_hybrid"),
        lambda_specific: float = default("objective.lambda_specific"),
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes != 2:
            raise ValueError("num_classes must be 2 for single-logit binary classification.")
        for name, value in (("lambda_cls", lambda_cls), ("lambda_intra", lambda_intra),
                            ("lambda_cross", lambda_cross), ("lambda_hybrid", lambda_hybrid),
                            ("lambda_specific", lambda_specific)):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            setattr(self, name, value)
        self.proxy_metric = ProxyMetricLearning(
            embedding_dim=embedding_dim, num_classes=num_classes,
            tau_intra=tau_intra, tau_cross=tau_cross,
            tau_hybrid=tau_hybrid, tau_specific=tau_specific,
        )

    def forward(self, logits: Tensor, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """Return six scalar losses from logits [B], embeddings [B,D], labels [B].

        Labels must be torch.long in {0,1}; the validated proxy module validates
        labels and embeddings. BCE targets are converted only after validation,
        to logits' floating dtype. All tensors/module parameters share device
        and floating dtype. No component is detached or skipped for zero weight.
        """
        if not isinstance(logits, Tensor) or logits.ndim != 1 or logits.shape[0] == 0:
            raise ValueError("logits must have shape [B] with B > 0.")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating-point.")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must not contain NaN or Inf.")
        if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must have shape [B], matching logits.")
        parameters = self.proxy_metric.hybrid_proxies
        if logits.device != parameters.device or logits.dtype != parameters.dtype:
            raise ValueError("logits and objective must share device and dtype.")
        proxy_losses = self.proxy_metric(z_mri, z_wsi, labels)
        classification = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))
        total = (self.lambda_cls * classification
                 + self.lambda_intra * proxy_losses["intra_modal"]
                 + self.lambda_cross * proxy_losses["cross_modal"]
                 + self.lambda_hybrid * proxy_losses["hybrid_proxy"]
                 + self.lambda_specific * proxy_losses["specific_proxy"])
        return {"total": total, "classification": classification, **proxy_losses}
