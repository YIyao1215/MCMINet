"""Cached-feature training composition of unchanged MRI, GAT and classifier."""
from torch import nn
from mcminet.models.cached_wsi_branch import CachedWSIBranch


class CachedBatchedMCMINet(nn.Module):
    """Reuse exact child objects from a canonical MCMINet/BatchedMCMINet state.

    WSI patch encoder remains OUTSIDE this registered computational graph.
    Apply training policy explicitly before constructing this execution wrapper.
    """
    def __init__(self,canonical):
        super().__init__()
        self.mri_branch=canonical.mri_branch
        self.wsi_branch=CachedWSIBranch(canonical.wsi_branch.gat_encoder, graph_config=dict(
            k=canonical.wsi_branch.graph_k, max_distance=canonical.wsi_branch.graph_max_distance,
            bidirectional=canonical.wsi_branch.graph_bidirectional))
        from copy import deepcopy
        self.candidate_config = deepcopy(getattr(canonical, 'candidate_config', None))
        self.smoke_test = getattr(canonical, 'smoke_test', False)
        self.encoder_initialization = dict(
            mri_pretrained=canonical.mri_branch.roi_encoder.pretrained,
            wsi_pretrained=canonical.wsi_branch.patch_encoder.pretrained)
        self.classifier=canonical.classifier

    def forward(self,tumor_roi,peritumoral_roi,lymph_node_roi,wsi_features_list,wsi_coordinates_list,
                return_embeddings=False,return_wsi_details=False,return_attention=False):
        if any(r.ndim!=4 for r in (tumor_roi,peritumoral_roi,lymph_node_roi)) or tumor_roi.shape[0]!=len(wsi_features_list):
            raise ValueError('MRI batch and WSI patient count must match')
        z_mri=self.mri_branch(tumor_roi,peritumoral_roi,lymph_node_roi)
        wsi=self.wsi_branch(wsi_features_list,wsi_coordinates_list,
            return_details=return_wsi_details,return_attention=return_attention)
        z_wsi=wsi['embedding'] if return_wsi_details or return_attention else wsi
        logits=self.classifier(z_mri,z_wsi)
        if not (return_embeddings or return_wsi_details or return_attention):return logits
        out=dict(logits=logits,z_mri=z_mri,z_wsi=z_wsi)
        if return_wsi_details or return_attention:out['wsi_output']=wsi
        return out
