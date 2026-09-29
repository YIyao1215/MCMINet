"""Patient-batched WSI encoding for variable-size slide graphs."""
from mcminet.config import default


from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mcminet.data.wsi_graph_builder import build_spatial_graph
from mcminet.models.wsi_patch_encoder import WSIPatchEncoder
from mcminet.models.wsi_gat_encoder import WSIGATEncoder


class BatchedWSIBranch(nn.Module):
    """Encode all patches jointly; construct spatial graphs per patient.

    One shared patch encoder and one shared GAT process each batch. Independent
    graphs are offset and joined as a disconnected graph, so no cross-patient
    edges are created. BatchNorm retains its validated train/eval behavior;
    disconnected topology does not imply independent batch statistics.
    Graph defaults match WSIBranch. Trainable controls the patch encoder only.
    A total of one node requires eval mode for the existing GAT BatchNorm.
    """

    def __init__(self, pretrained: bool = default("WSI.pretrained"), trainable: bool = True,
                 graph_k: int = default("graph.k"), graph_max_distance: float | None = default("graph.max_distance"),
                 graph_bidirectional: bool = default("graph.bidirectional")) -> None:
        super().__init__()
        self.patch_encoder = WSIPatchEncoder(pretrained=pretrained, trainable=trainable)
        self.gat_encoder = WSIGATEncoder(input_dim=default("WSI.node_feature_dim"))
        self.graph_k = graph_k
        self.graph_max_distance = graph_max_distance
        self.graph_bidirectional = graph_bidirectional

    def forward(self, patches_list: Sequence[Tensor], coordinates_list: Sequence[Tensor],
                return_details: bool = False, return_attention: bool = False) -> Tensor | dict[str, Any]:
        """Accept patient sequences [Ni,3,H,W] and [Ni,2], preserving order.

        Ni must be positive and may vary. All patches must share spatial size,
        floating dtype and device; no resize/cast occurs. Coordinates are real
        tensors in each slide's local units, validated by the graph builder.
        Default returns normalized embeddings [B,256]. Either return flag gives
        details, including ResNet node_features [sum(Ni),512], full gat_output,
        attention, batch [sum(Ni)], and patch_counts [B]. Attention uses global
        node indices; no patient-level interpretation is performed here.
        edge_distance stays CPU float64 as in WSIBranch; edges, counts and batch
        use the feature device. Coordinates are neither shifted nor modified.
        """
        for name, values in (("patches_list", patches_list), ("coordinates_list", coordinates_list)):
            if isinstance(values, (Tensor, str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence of patient tensors, not a Tensor.")
        if not patches_list or len(patches_list) != len(coordinates_list):
            raise ValueError("Patient sequences must be nonempty and have matching lengths.")
        counts = []
        for patches, coordinates in zip(patches_list, coordinates_list):
            if not isinstance(patches, Tensor) or patches.ndim != 4 or patches.shape[1] != 3:
                raise ValueError("Each patient's patches must have shape [Ni,3,H,W].")
            if patches.shape[0] == 0 or patches.shape[2] == 0 or patches.shape[3] == 0:
                raise ValueError("Patient patch count and spatial dimensions must be nonempty.")
            if not patches.is_floating_point():
                raise TypeError("Patches must be floating-point.")
            first = patches_list[0]
            if patches.shape[1:] != first.shape[1:]:
                raise ValueError("All patients must have identical patch spatial sizes.")
            if patches.device != first.device or patches.dtype != first.dtype:
                raise ValueError("All patches must share device and dtype.")
            if not torch.isfinite(patches).all():
                raise ValueError("Patches must not contain NaN or Inf.")
            if not isinstance(coordinates, Tensor) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError("Each patient's coordinates must be a tensor shaped [Ni,2].")
            if coordinates.shape[0] != patches.shape[0]:
                raise ValueError("Patient patch and coordinate counts must match.")
            counts.append(patches.shape[0])

        edge_parts, distance_parts = [], []
        offset = 0
        for coordinates, count in zip(coordinates_list, counts):
            graph = build_spatial_graph(coordinates, k=self.graph_k,
                                        max_distance=self.graph_max_distance,
                                        bidirectional=self.graph_bidirectional)
            edge_parts.append(graph["edge_index"] + offset)
            distance_parts.append(graph["edge_distance"])
            offset += count
        node_features = self.patch_encoder(torch.cat(tuple(patches_list), dim=0))
        edges = torch.cat(edge_parts, dim=1).to(node_features.device)
        distances = torch.cat(distance_parts)
        patch_counts = torch.tensor(counts, dtype=torch.long, device=node_features.device)
        batch = torch.repeat_interleave(torch.arange(len(counts), device=node_features.device), patch_counts)
        # Validated GAT exposes attention as an instance option, not a forward
        # argument. Restore it even on failure; no GAT implementation is copied.
        previous = self.gat_encoder.return_attention
        try:
            self.gat_encoder.return_attention = return_attention
            gat_output = self.gat_encoder(node_features, edges, batch)
        finally:
            self.gat_encoder.return_attention = previous
        if not return_details and not return_attention:
            return gat_output["embedding"]
        return {"embedding": gat_output["embedding"], "node_features": node_features,
                "edge_index": edges, "edge_distance": distances, "batch": batch,
                "patch_counts": patch_counts, "attention": gat_output["attention"],
                "gat_output": gat_output}
