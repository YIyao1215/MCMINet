"""Three-ROI MRI patient representation using a single shared ROI encoder."""
from mcminet.config import default


from math import isfinite
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mcminet.models.mri_roi_encoder import MRIROIEncoder


class MRIBranch(nn.Module):
    """Encode tumor, peritumoral and lymph_node ROIs with shared weights.

    Default dimensions: three 256-D ROI embeddings -> concatenate to 768-D
    -> Linear(768, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 256)
    -> L2-normalized patient embedding. No classification or preprocessing.

    ``trainable`` controls the shared ROI encoder, not the fusion MLP.
    Freezing parameters does not force eval mode; BatchNorm running statistics
    continue to follow train()/eval(). ``pretrained`` is forwarded to the ROI
    encoder and may download uncached weights when True.
    """

    def __init__(
        self, roi_embedding_dim: int = default("MRI.roi_embedding_dim"), fusion_hidden_dim: int = default("MRI.fusion_hidden_dim"),
        output_dim: int = default("MRI.patient_embedding_dim"), dropout: float = default("MRI.fusion_dropout"),
        pretrained: bool = default("MRI.pretrained"), trainable: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("roi_embedding_dim", roi_embedding_dim),
                            ("fusion_hidden_dim", fusion_hidden_dim), ("output_dim", output_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (isinstance(dropout, bool) or not isinstance(dropout, Real)
                or not isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.roi_encoder = MRIROIEncoder(
            embedding_dim=roi_embedding_dim, pretrained=pretrained, trainable=trainable,
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * roi_embedding_dim, fusion_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(
        self, tumor_roi: Tensor, peritumoral_roi: Tensor, lymph_node_roi: Tensor,
        return_roi_embeddings: bool = False,
    ) -> Tensor | dict[str, Tensor | dict[str, Tensor]]:
        """Accept three floating [B, 1, H, W] ROIs with equal nonzero B.

        Spatial sizes may differ. The ROI encoder validates channels, floating
        dtype and spatial extent; this branch validates batch consistency.
        Returns [B, output_dim], or an ``embedding`` plus ``roi_embeddings``
        dictionary with tumor/peritumoral/lymph_node [B, roi_embedding_dim].
        All inputs must match model device/dtype; no implicit transfer occurs.
        """
        inputs = (tumor_roi, peritumoral_roi, lymph_node_roi)
        for name, roi in zip(("tumor", "peritumoral", "lymph_node"), inputs):
            if not isinstance(roi, Tensor) or roi.ndim != 4:
                raise ValueError(f"{name} ROI must be a tensor shaped [B, 1, H, W].")
            if roi.shape[0] == 0:
                raise ValueError(f"{name} ROI batch must be nonempty.")
        if len({roi.shape[0] for roi in inputs}) != 1:
            raise ValueError("All three ROI batch sizes must match.")

        z_tumor = self.roi_encoder(tumor_roi)
        z_peritumoral = self.roi_encoder(peritumoral_roi)
        z_lymph_node = self.roi_encoder(lymph_node_roi)
        concatenated = torch.cat((z_tumor, z_peritumoral, z_lymph_node), dim=1)
        embedding = F.normalize(self.fusion(concatenated), p=2, dim=1)
        if return_roi_embeddings:
            return {"embedding": embedding, "roi_embeddings": {
                "tumor": z_tumor, "peritumoral": z_peritumoral, "lymph_node": z_lymph_node,
            }}
        return embedding
