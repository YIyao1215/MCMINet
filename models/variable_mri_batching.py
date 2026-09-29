"""Native MRI batching with exact shared validated branch math and explicit BN policy."""
from collections.abc import Sequence
from pathlib import Path
import hashlib
import torch
from mcminet.models.cached_batched_mcminet import CachedBatchedMCMINet
from mcminet.training.mri_batchnorm_policy import (
    MRI_BN_POLICY, apply_mri_batchnorm_policy, audit_mri_batchnorm_policy,
)

BATCHING_POLICY = "native_patientwise_validated_mri_forward_then_feature_concat_v1"


def policy_fingerprint():
    root = Path(__file__).resolve().parents[1]
    names = ("models/variable_mri_batching.py", "training/mri_batchnorm_policy.py",
             "data/variable_mri_collate.py", "training/native_mri_checkpoint.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _validate_lists(tumor, peritumoral, lymph_node):
    values = (tumor, peritumoral, lymph_node)
    for items in values:
        if isinstance(items, (torch.Tensor, str, bytes)) or not isinstance(items, Sequence):
            raise ValueError("Native MRI inputs must be patient sequences")
    if not tumor or len({len(items) for items in values}) != 1:
        raise ValueError("Nonempty matching MRI patient lists required")
    for items in values:
        for roi in items:
            if not isinstance(roi, torch.Tensor) or roi.ndim != 3 or roi.shape[0] != 1 or min(roi.shape) < 1:
                raise ValueError("Native MRI ROI must be nonempty [1,H,W]")
            if roi.dtype != torch.float32 or not torch.isfinite(roi).all():
                raise ValueError("Native MRI ROI must be finite float32")


def encode_variable_mri(mri_branch, tumor, peritumoral, lymph_node, *, return_roi_embeddings=False):
    """Use the SAME MRIBranch and its one MRIROIEncoder for all 3B inputs.

    The original branch performs the three ROI projections, fusion and final L2
    normalization independently for each patient. Only [1,256] results are joined.
    Unsqueeze/to(device) add a batch axis or copy device; no spatial transform.
    """
    _validate_lists(tumor, peritumoral, lymph_node)
    audit_mri_batchnorm_policy(mri_branch)
    device = next(mri_branch.parameters()).device
    patient_outputs = []
    roi_outputs = {name: [] for name in ("tumor", "peritumoral", "lymph_node")}
    for patient_rois in zip(tumor, peritumoral, lymph_node):
        output = mri_branch(*(roi.unsqueeze(0).to(device) for roi in patient_rois),
                            return_roi_embeddings=return_roi_embeddings)
        if return_roi_embeddings:
            patient_outputs.append(output["embedding"])
            for name in roi_outputs:
                roi_outputs[name].append(output["roi_embeddings"][name])
        else:
            patient_outputs.append(output)
    embedding = torch.cat(patient_outputs, dim=0)
    if not return_roi_embeddings:
        return embedding
    return dict(embedding=embedding,
                roi_embeddings={name: torch.cat(items, dim=0) for name, items in roi_outputs.items()})


class VariableMRICachedMCMINet(CachedBatchedMCMINet):
    """No new trainable tensors, no copied encoders, unchanged state_dict names.

    Different repr/policy identity intentionally prevents silent cross-policy
    release resume. The validated checkpoint schema supports same-policy resume.
    Stacked tensors are also accepted for compatibility with the validated 6B
    runner on its existing stackable-input domain.
    """
    def __init__(self, canonical):
        super().__init__(canonical)
        self._native_policy_sha256 = policy_fingerprint()
        self.train(canonical.training)

    def extra_repr(self):
        return (f"batching_policy={BATCHING_POLICY}, mri_bn_policy={MRI_BN_POLICY}, "
                f"policy_sha256={self._native_policy_sha256}")

    def train(self, mode=True):
        super().train(mode)
        apply_mri_batchnorm_policy(self.mri_branch)
        return self

    def forward(self, tumor_roi, peritumoral_roi, lymph_node_roi,
                wsi_features_list, wsi_coordinates_list, return_embeddings=False,
                return_wsi_details=False, return_attention=False):
        audit_mri_batchnorm_policy(self.mri_branch)
        # The old same-size data interface remains usable with this MRI BN policy.
        if all(isinstance(x, torch.Tensor) for x in (tumor_roi, peritumoral_roi, lymph_node_roi)):
            return super().forward(tumor_roi, peritumoral_roi, lymph_node_roi,
                                   wsi_features_list, wsi_coordinates_list,
                                   return_embeddings, return_wsi_details, return_attention)
        _validate_lists(tumor_roi, peritumoral_roi, lymph_node_roi)
        if len(tumor_roi) != len(wsi_features_list) or len(tumor_roi) != len(wsi_coordinates_list):
            raise ValueError("MRI and WSI patient counts must match")
        z_mri = encode_variable_mri(self.mri_branch, tumor_roi, peritumoral_roi, lymph_node_roi)
        wsi = self.wsi_branch(wsi_features_list, wsi_coordinates_list,
                              return_details=return_wsi_details, return_attention=return_attention)
        z_wsi = wsi["embedding"] if return_wsi_details or return_attention else wsi
        logits = self.classifier(z_mri, z_wsi)
        if not (return_embeddings or return_wsi_details or return_attention):
            return logits
        result = dict(logits=logits, z_mri=z_mri, z_wsi=z_wsi)
        if return_wsi_details or return_attention:
            result["wsi_output"] = wsi
        return result
