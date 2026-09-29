"""Compose patch encoding, spatial adjacency, and GAT encoding for one patient."""
from mcminet.config import default


import numpy as np
import torch
from torch import Tensor, nn

from mcminet.data.wsi_graph_builder import build_spatial_graph
from mcminet.models.wsi_gat_encoder import WSIGATEncoder
from mcminet.models.wsi_patch_encoder import WSIPatchEncoder


class WSIBranch(nn.Module):
    """Encode one patient's normalized RGB patches into a 256-D embedding.

    Spatial connectivity uses only supplied patch centers, in caller-defined
    units. No final experimental distance threshold is assumed. ``pretrained``
    and ``trainable`` control the existing patch encoder; ``return_attention``
    controls the existing GAT's raw attention records. Pretrained construction
    may download uncached weights. Single-node graphs require eval mode because
    the validated GAT uses BatchNorm; this wrapper does not change module modes.
    """

    def __init__(
        self,
        pretrained: bool = default("WSI.pretrained"),
        trainable: bool = True,
        graph_k: int = default("graph.k"),
        graph_max_distance: float | None = default("graph.max_distance"),
        graph_bidirectional: bool = default("graph.bidirectional"),
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.patch_encoder = WSIPatchEncoder(pretrained=pretrained, trainable=trainable)
        self.gat_encoder = WSIGATEncoder(input_dim=default("WSI.node_feature_dim"), return_attention=return_attention)
        self.graph_k = graph_k
        self.graph_max_distance = graph_max_distance
        self.graph_bidirectional = graph_bidirectional

    def forward(self, patches: Tensor, coordinates: Tensor | np.ndarray) -> dict:
        """Map patches [N, 3, H, W] and aligned centers [N, 2] to one graph.

        Returns ``embedding`` [1, 256], ResNet ``node_features`` [N, 512],
        spatial ``edge_index`` [2, E], and CPU float64 ``edge_distance`` [E].
        ``gat_output`` preserves the full GAT dictionary: embedding, pooled
        features [1, 64], final node features [N, 64], and optional attention.
        Spatial edges omit self-loops; GAT attention records may include them.

        Patches are never moved implicitly. The caller must place the model
        and patches on matching devices. Spatial edges and the single-graph
        batch vector are explicitly placed on the extracted features' device.
        Distances remain on CPU to preserve float64 precision, including when
        model execution uses a device without float64 support.
        """
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape [N, 3, H, W].")
        if not isinstance(coordinates, (Tensor, np.ndarray)):
            raise TypeError("coordinates must be a torch.Tensor or numpy.ndarray.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape [N, 2].")
        if patches.shape[0] != coordinates.shape[0]:
            raise ValueError("Number of patches must equal number of coordinate rows.")
        if patches.shape[0] == 0:
            raise ValueError("At least one patch is required for a patient embedding.")

        graph = build_spatial_graph(
            coordinates, k=self.graph_k, max_distance=self.graph_max_distance,
            bidirectional=self.graph_bidirectional,
        )
        node_features = self.patch_encoder(patches)
        edge_index = graph["edge_index"].to(device=node_features.device)
        batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device)
        gat_output = self.gat_encoder(node_features, edge_index, batch)
        return {
            "embedding": gat_output["embedding"],
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_distance": graph["edge_distance"],
            "gat_output": gat_output,
        }
