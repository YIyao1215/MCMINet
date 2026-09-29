"""Thin multi-patient composition of the validated MRI, WSI and classifier modules."""
from mcminet.config import default


from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from mcminet.models.mri_branch import MRIBranch
from mcminet.models.batched_wsi_branch import BatchedWSIBranch
from mcminet.models.multimodal_classifier import MultimodalClassifier


class BatchedMCMINet(nn.Module):
    """Fuse MRI rows and variable-size WSI graphs in caller-supplied order.

    Row/list item i must belong to the same patient; alignment is the caller's
    responsibility. No sorting, normalization, loss or preprocessing is added.
    Embedding width is fixed at 256 by the validated WSI branch.
    Pretrained flags may download weights; trainable flags control only the
    MRI ROI encoder / WSI patch encoder, retaining child mode semantics.
    """

    def __init__(
        self, embedding_dim: int = default("model.embedding_dim"), mri_roi_embedding_dim: int = default("MRI.roi_embedding_dim"),
        mri_fusion_hidden_dim: int = default("MRI.fusion_hidden_dim"), mri_dropout: float = default("MRI.fusion_dropout"),
        classifier_hidden_dim: int = default("model.classifier_hidden_dim"), classifier_dropout: float = default("model.classifier_dropout"),
        mri_pretrained: bool = default("MRI.pretrained"), mri_trainable: bool = True,
        wsi_pretrained: bool = default("WSI.pretrained"), wsi_trainable: bool = True,
        wsi_graph_k: int = default("graph.k"), wsi_graph_max_distance: float | None = default("graph.max_distance"),
        wsi_graph_bidirectional: bool = default("graph.bidirectional"),
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim != default("WSI.patient_embedding_dim"):
            raise ValueError("embedding_dim must be 256 to match the validated BatchedWSIBranch.")
        self.embedding_dim = embedding_dim
        self.mri_branch = MRIBranch(
            roi_embedding_dim=mri_roi_embedding_dim, fusion_hidden_dim=mri_fusion_hidden_dim,
            output_dim=embedding_dim, dropout=mri_dropout,
            pretrained=mri_pretrained, trainable=mri_trainable,
        )
        self.wsi_branch = BatchedWSIBranch(
            pretrained=wsi_pretrained, trainable=wsi_trainable,
            graph_k=wsi_graph_k, graph_max_distance=wsi_graph_max_distance,
            graph_bidirectional=wsi_graph_bidirectional,
        )
        self.classifier = MultimodalClassifier(
            embedding_dim=embedding_dim, hidden_dim=classifier_hidden_dim, dropout=classifier_dropout,
        )

    def forward(
        self, tumor_roi: Tensor, peritumoral_roi: Tensor, lymph_node_roi: Tensor,
        wsi_patches_list: Sequence[Tensor], wsi_coordinates_list: Sequence[Tensor],
        return_embeddings: bool = False, return_wsi_details: bool = False,
        return_attention: bool = False,
    ) -> Tensor | dict[str, Any]:
        """Accept MRI [B,1,H_i,W_i], patch lists [Ni,3,H,W], centers [Ni,2].

        B must be positive and match both sequence lengths. MRI spatial sizes
        may differ. Lower-level input validation remains with child modules.
        Default returns raw logits [B]. Any return flag yields logits and
        autograd-preserving z_mri/z_wsi [B,256]. Details or attention additionally
        include the original WSI dictionary; attention implies WSI details.
        """
        rois = (tumor_roi, peritumoral_roi, lymph_node_roi)
        if any(not isinstance(roi, Tensor) or roi.ndim != 4 for roi in rois):
            raise ValueError("MRI ROIs must be tensors shaped [B, 1, H, W].")
        if len({roi.shape[0] for roi in rois}) != 1:
            raise ValueError("MRI ROI batch sizes must match.")
        batch_size = tumor_roi.shape[0]
        if batch_size == 0:
            raise ValueError("MRI batch must be nonempty (B > 0).")
        if len(wsi_patches_list) != batch_size or len(wsi_coordinates_list) != batch_size:
            raise ValueError("MRI batch size must match the number of WSI patient graphs.")

        z_mri = self.mri_branch(tumor_roi, peritumoral_roi, lymph_node_roi)
        need_wsi_details = return_wsi_details or return_attention
        wsi_output = self.wsi_branch(
            wsi_patches_list, wsi_coordinates_list,
            return_details=need_wsi_details, return_attention=return_attention,
        )
        z_wsi = wsi_output["embedding"] if need_wsi_details else wsi_output
        for name, embedding in (("MRI", z_mri), ("WSI", z_wsi)):
            if not isinstance(embedding, Tensor) or embedding.shape != (batch_size, self.embedding_dim):
                raise ValueError(
                    f"{name} branch must return embedding shape [{batch_size}, {self.embedding_dim}]."
                )
        logits = self.classifier(z_mri, z_wsi)
        if not isinstance(logits, Tensor) or logits.shape != (batch_size,):
            raise ValueError(f"Classifier must return logits shape [{batch_size}].")
        if not return_embeddings and not need_wsi_details:
            return logits
        output = {"logits": logits, "z_mri": z_mri, "z_wsi": z_wsi}
        if need_wsi_details:
            output["wsi_output"] = wsi_output
        return output
