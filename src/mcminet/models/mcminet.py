"""Thin single-patient composition of the validated MRI, WSI and classifier modules."""
from mcminet.config import default


from typing import Any

import numpy as np
from torch import Tensor, nn

from mcminet.models.mri_branch import MRIBranch
from mcminet.models.wsi_branch import WSIBranch
from mcminet.models.multimodal_classifier import MultimodalClassifier


class MCMINet(nn.Module):
    """Return raw logits from three MRI ROIs and one WSI patch graph.

    This wrapper currently supports one patient per forward pass to preserve
    compatibility with the validated single-slide WSI graph branch. Multi-patient
    graph batching will be addressed separately in the training data pipeline;
    it is required for intra-modal patient negatives during real training.
    The validated WSI branch fixes the shared embedding width at 256.

    Pretrained flags may download uncached weights. Trainable flags retain the
    child modules' semantics: MRI ROI encoder / WSI patch encoder parameters
    only, without forcing eval mode or freezing fusion/GAT/classifier layers.
    No loss, probabilities, or labels are part of this model API.
    """

    def __init__(
        self, embedding_dim: int = default("model.embedding_dim"), mri_roi_embedding_dim: int = default("MRI.roi_embedding_dim"),
        mri_fusion_hidden_dim: int = default("MRI.fusion_hidden_dim"), mri_dropout: float = default("MRI.fusion_dropout"),
        classifier_hidden_dim: int = default("model.classifier_hidden_dim"), classifier_dropout: float = default("model.classifier_dropout"),
        mri_pretrained: bool = default("MRI.pretrained"), mri_trainable: bool = True,
        wsi_pretrained: bool = default("WSI.pretrained"), wsi_trainable: bool = True,
        wsi_graph_k: int = default("graph.k"), wsi_graph_max_distance: float | None = default("graph.max_distance"),
        wsi_graph_bidirectional: bool = default("graph.bidirectional"), wsi_return_attention: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim != default("WSI.patient_embedding_dim"):
            raise ValueError("embedding_dim must be 256 to match the validated WSIBranch.")
        self.embedding_dim = embedding_dim
        self.mri_branch = MRIBranch(
            roi_embedding_dim=mri_roi_embedding_dim, fusion_hidden_dim=mri_fusion_hidden_dim,
            output_dim=embedding_dim, dropout=mri_dropout,
            pretrained=mri_pretrained, trainable=mri_trainable,
        )
        self.wsi_branch = WSIBranch(
            pretrained=wsi_pretrained, trainable=wsi_trainable,
            graph_k=wsi_graph_k, graph_max_distance=wsi_graph_max_distance,
            graph_bidirectional=wsi_graph_bidirectional, return_attention=wsi_return_attention,
        )
        self.classifier = MultimodalClassifier(
            embedding_dim=embedding_dim, hidden_dim=classifier_hidden_dim, dropout=classifier_dropout,
        )

    def forward(
        self, tumor_roi: Tensor, peritumoral_roi: Tensor, lymph_node_roi: Tensor,
        wsi_patches: Tensor, wsi_coordinates: Tensor | np.ndarray,
        return_embeddings: bool = False, return_wsi_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        """Accept MRI [1,1,H_i,W_i], WSI patches [N,3,H,W], centers [N,2].

        Default output is raw logits [1]. Either return flag yields a dictionary
        with logits, z_mri [1,256], z_wsi [1,256]; return_wsi_details additionally
        includes the unchanged full WSIBranch dictionary under wsi_output.
        Embeddings retain autograd; the wrapper does not normalize or move them.
        Detailed input validation and mode behavior remain with child modules.
        """
        rois = (tumor_roi, peritumoral_roi, lymph_node_roi)
        if any(not isinstance(roi, Tensor) or roi.ndim != 4 for roi in rois):
            raise ValueError("MRI ROIs must be tensors shaped [B, 1, H, W].")
        if len({roi.shape[0] for roi in rois}) != 1:
            raise ValueError("MRI ROI batch sizes must match.")
        if tumor_roi.shape[0] != 1:
            raise ValueError(
                "Current MCMINet wrapper supports one patient per forward pass because "
                "the validated WSI branch currently operates on a single-slide graph."
            )
        z_mri = self.mri_branch(tumor_roi, peritumoral_roi, lymph_node_roi)
        wsi_output = self.wsi_branch(wsi_patches, wsi_coordinates)
        z_wsi = wsi_output["embedding"]
        for name, embedding in (("MRI", z_mri), ("WSI", z_wsi)):
            if embedding.shape != (1, self.embedding_dim):
                raise ValueError(f"{name} branch must return embedding shape [1, {self.embedding_dim}].")
        logits = self.classifier(z_mri, z_wsi)
        if not return_embeddings and not return_wsi_details:
            return logits
        output = {"logits": logits, "z_mri": z_mri, "z_wsi": z_wsi}
        if return_wsi_details:
            output["wsi_output"] = wsi_output
        return output
