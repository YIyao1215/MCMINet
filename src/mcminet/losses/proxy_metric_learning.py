"""Proxy-assisted metric learning with four separate, unweighted losses."""
from mcminet.config import default


from math import isfinite
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProxyMetricLearning(nn.Module):
    """Learn shared hybrid and modality-specific class proxies.

    Each proxy bank is [num_classes, embedding_dim]. Embeddings and proxies
    are functionally L2-normalized before cosine similarities; raw parameters
    are never overwritten or detached. Default labels are 0=Non-responder,
    1=Responder; additional classes may be configured. Inputs and parameters
    must share floating dtype/device, with torch.long labels on that device.
    No direct MRI–WSI patient-pair matching or total objective is computed.
    """

    def __init__(
        self, embedding_dim: int = default("model.embedding_dim"), num_classes: int = default("model.num_classes"),
        tau_intra: float = default("objective.tau_intra"), tau_cross: float = default("objective.tau_cross"),
        tau_hybrid: float = default("objective.tau_hybrid"), tau_specific: float = default("objective.tau_specific"),
    ) -> None:
        super().__init__()
        for name, value, minimum in (("embedding_dim", embedding_dim, 1), ("num_classes", num_classes, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name, value in (("tau_intra", tau_intra), ("tau_cross", tau_cross),
                            ("tau_hybrid", tau_hybrid), ("tau_specific", tau_specific)):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
            setattr(self, name, value)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hybrid_proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.mri_proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.wsi_proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def _normalized_inputs(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        for name, value in (("z_mri", z_mri), ("z_wsi", z_wsi)):
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != self.embedding_dim:
                raise ValueError(f"{name} must have shape [B, {self.embedding_dim}].")
            if value.shape[0] == 0:
                raise ValueError("Batch must be nonempty.")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating-point.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must not contain NaN or Inf.")
            if value.device != self.hybrid_proxies.device or value.dtype != self.hybrid_proxies.dtype:
                raise ValueError("Embeddings and module parameters must share device and dtype.")
        if z_mri.shape[0] != z_wsi.shape[0]:
            raise ValueError("MRI and WSI batch sizes must match.")
        if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.shape[0] != z_mri.shape[0]:
            raise ValueError("labels must have shape [B], matching embeddings.")
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long.")
        if labels.device != z_mri.device:
            raise ValueError("labels and embeddings must share a device.")
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError("labels must be within [0, num_classes).")
        return F.normalize(z_mri, p=2, dim=1), F.normalize(z_wsi, p=2, dim=1)

    def _intra_contributions(self, z: Tensor, hybrid: Tensor, labels: Tensor) -> Tensor:
        """Per-valid-anchor losses for ONE modality; inputs already normalized."""
        negatives = labels[:, None] != labels[None, :]
        valid = negatives.any(dim=1)
        positive = (z * hybrid[labels]).sum(dim=1) / self.tau_intra
        sample_logits = (z @ z.T / self.tau_intra).masked_fill(~negatives, -torch.inf)
        logits = torch.cat((positive[:, None], sample_logits), dim=1)
        return (torch.logsumexp(logits, dim=1) - positive)[valid]

    def compute_intra_modal_loss(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> Tensor:
        """Hybrid true-class positive versus all same-modality wrong-class samples.

        Only valid anchor contributions are pooled across both modalities.
        With no negatives, return a scalar autograd-compatible zero.
        """
        mri, wsi = self._normalized_inputs(z_mri, z_wsi, labels)
        hybrid = F.normalize(self.hybrid_proxies, p=2, dim=1)
        contributions = torch.cat((self._intra_contributions(mri, hybrid, labels),
                                   self._intra_contributions(wsi, hybrid, labels)))
        if contributions.numel() == 0:
            return (mri.sum() + wsi.sum() + hybrid.sum()) * 0.0
        return contributions.mean()

    @staticmethod
    def _proxy_ce(z: Tensor, proxies: Tensor, labels: Tensor, tau: float) -> Tensor:
        return F.cross_entropy(z @ F.normalize(proxies, p=2, dim=1).T / tau, labels)

    def compute_cross_modal_loss(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> Tensor:
        """MRI -> WSI proxies and WSI -> MRI proxies, equally averaged CE."""
        mri, wsi = self._normalized_inputs(z_mri, z_wsi, labels)
        return 0.5 * (self._proxy_ce(mri, self.wsi_proxies, labels, self.tau_cross)
                      + self._proxy_ce(wsi, self.mri_proxies, labels, self.tau_cross))

    def compute_hybrid_proxy_loss(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> Tensor:
        """Both modalities classify against the same hybrid class proxies."""
        mri, wsi = self._normalized_inputs(z_mri, z_wsi, labels)
        return 0.5 * (self._proxy_ce(mri, self.hybrid_proxies, labels, self.tau_hybrid)
                      + self._proxy_ce(wsi, self.hybrid_proxies, labels, self.tau_hybrid))

    def compute_specific_proxy_loss(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> Tensor:
        """MRI -> MRI proxies and WSI -> WSI proxies, equally averaged CE."""
        mri, wsi = self._normalized_inputs(z_mri, z_wsi, labels)
        return 0.5 * (self._proxy_ce(mri, self.mri_proxies, labels, self.tau_specific)
                      + self._proxy_ce(wsi, self.wsi_proxies, labels, self.tau_specific))

    def forward(self, z_mri: Tensor, z_wsi: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """Return four independent scalar losses; no weights or total loss."""
        return {
            "intra_modal": self.compute_intra_modal_loss(z_mri, z_wsi, labels),
            "cross_modal": self.compute_cross_modal_loss(z_mri, z_wsi, labels),
            "hybrid_proxy": self.compute_hybrid_proxy_loss(z_mri, z_wsi, labels),
            "specific_proxy": self.compute_specific_proxy_loss(z_mri, z_wsi, labels),
        }
