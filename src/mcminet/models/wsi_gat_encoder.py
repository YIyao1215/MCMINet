"""Two-layer graph attention encoder for precomputed WSI patch features."""
from mcminet.config import default


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class WSIGATEncoder(nn.Module):
    """Encode one graph per patient into an L2-normalized WSI embedding.

    ``input_dim`` is the supplied patch feature width. ``gat1_dim`` and
    ``gat2_dim`` are widths per attention head; heads are concatenated.
    ``dropout`` applies to GAT attention coefficients. With defaults, node
    widths are input_dim -> 128 -> 64, followed by pooling and a 256-D
    projection. Batch normalization requires at least two total nodes when
    training (a single-node graph can be part of a larger batch).

    Attention is returned only when ``return_attention`` is enabled. Records
    preserve each layer's edges (including GATConv's default self-loops) and
    per-head coefficients without aggregation. Returned attention tensors
    are detached to avoid retaining the autograd graph for interpretation.
    """

    def __init__(
        self,
        input_dim: int,
        gat1_dim: int = default("WSI.gat1_dim"),
        gat2_dim: int = default("WSI.gat2_dim"),
        heads: int = default("WSI.gat_heads"),
        embedding_dim: int = default("WSI.patient_embedding_dim"),
        dropout: float = default("WSI.gat_dropout"),
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("gat1_dim", gat1_dim),
            ("gat2_dim", gat2_dim),
            ("heads", heads),
            ("embedding_dim", embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.input_dim = input_dim
        self.return_attention = return_attention
        self.gat1 = GATConv(input_dim, gat1_dim, heads=heads, concat=True,
                            dropout=dropout)
        self.bn1 = nn.BatchNorm1d(gat1_dim * heads)
        self.gat2 = GATConv(gat1_dim * heads, gat2_dim, heads=heads,
                            concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(gat2_dim * heads)
        self.projection = nn.Linear(gat2_dim * heads, embedding_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> dict:
        """Encode a batch of nonempty patient graphs.

        Args:
            x: Floating-point patch features with shape [N, input_dim].
            edge_index: Long tensor of source/target node indices, [2, E].
                Indices must be in [0, N); edges must stay within each graph.
            batch: Long tensor [N] assigning nodes to graphs, with contiguous
                graph IDs starting at zero. All inputs must share a device.

        Returns:
            Dictionary containing ``embedding`` [B, embedding_dim],
            ``pooled_features`` [B, gat2_dim * heads], ``node_features``
            [N, gat2_dim * heads], and ``attention`` (None or a dictionary
            for gat1 and gat2). Each attention record contains ``edge_index``
            [2, E_layer] and unaveraged ``alpha`` [E_layer, heads].
        """
        if x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] == 0:
            raise ValueError(f"x must have shape [N, {self.input_dim}] with N > 0.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError("batch must have shape [N], matching x.")
        if edge_index.dtype != torch.long or batch.dtype != torch.long:
            raise TypeError("edge_index and batch must be torch.long tensors.")
        if x.device != edge_index.device or x.device != batch.device:
            raise ValueError("x, edge_index, and batch must share a device.")

        attention = None
        if self.return_attention:
            x, (edges1, alpha1) = self.gat1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.gat1(x, edge_index)
        x = self.bn1(F.relu(x))  # [N, gat1_dim * heads], default [N, 128].

        if self.return_attention:
            x, (edges2, alpha2) = self.gat2(
                x, edge_index, return_attention_weights=True
            )
            attention = {
                "gat1": {"edge_index": edges1.detach(), "alpha": alpha1.detach()},
                "gat2": {"edge_index": edges2.detach(), "alpha": alpha2.detach()},
            }
        else:
            x = self.gat2(x, edge_index)
        node_features = self.bn2(F.relu(x))  # [N, 64] with defaults.
        pooled_features = global_mean_pool(node_features, batch)  # [B, 64].
        embedding = F.normalize(self.projection(pooled_features), p=2, dim=-1)
        return {
            "embedding": embedding,  # [B, 256] with defaults.
            "pooled_features": pooled_features,
            "node_features": node_features,
            "attention": attention,
        }
