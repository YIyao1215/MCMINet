"""Build spatial adjacency from WSI patch centers, independently of features."""
from mcminet.config import default


from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor


def build_spatial_graph(
    coordinates: Tensor | np.ndarray,
    k: int = default("graph.k"),
    max_distance: float | None = default("graph.max_distance"),
    bidirectional: bool = default("graph.bidirectional"),
) -> dict[str, Tensor]:
    """Construct a distance-constrained kNN graph of WSI patches.

    Each node represents one patch. For each node, select up to ``k`` nearest
    other nodes by Euclidean distance, then retain candidates whose distance
    is <= ``max_distance`` when supplied. Ties favor the lower node index.
    Edges represent local spatial adjacency only: patch features and pathology
    labels play no role, so adjacent tumor and stroma patches may connect.

    Args:
        coordinates: Real numeric tensor or NumPy array [N, 2] of patch-center
            [x, y] positions for ONE graph. Units are caller-defined. The planned
            WSI pipeline will convert centers to normalized patch-grid units,
            where orthogonal neighbors are distance 1 and diagonal neighbors
            are distance sqrt(2). This function does not perform that conversion.
        k: Positive integer limiting candidate neighbors per node before
            filtering and symmetrization. Uses min(k, N - 1) for N > 0.
        max_distance: Optional finite, nonnegative distance threshold in the
            supplied coordinate units. None gives standard kNN. Filtering may
            leave isolated patches; they are never force-connected.
        bidirectional: Add the reverse of every retained edge and deduplicate
            directed pairs. Final node degree may exceed k after this step.

    Returns:
        A dictionary with CPU tensors ``edge_index`` (torch.long, [2, E])
        and ``edge_distance`` (torch.float64, [E]). Column e denotes source
        -> target, with its Euclidean distance at position e. Edges are sorted
        by source then target and contain no self-loops. Empty inputs, singleton
        inputs, and graphs with no retained edges return [2, 0] and [0]. Node
        count must be retained by the caller, including isolated nodes.

    Notes:
        Coordinates are detached and converted to CPU float64; construction is
        not differentiable. Dense pairwise distances use O(N^2) memory, and
        stable neighbor sorting uses O(N^2 log N) time. This correctness-first
        utility is intended for graphs that fit in memory.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if max_distance is not None:
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, Real)
            or not np.isfinite(max_distance)
            or max_distance < 0
        ):
            raise ValueError("max_distance must be finite and nonnegative, or None.")
    if not isinstance(bidirectional, bool):
        raise TypeError("bidirectional must be a bool.")
    if not isinstance(coordinates, (Tensor, np.ndarray)):
        raise TypeError("coordinates must be a torch.Tensor or numpy.ndarray.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [N, 2].")
    if isinstance(coordinates, np.ndarray):
        if coordinates.dtype.kind not in "iuf":
            raise TypeError("coordinates must contain real numeric values.")
        # Copy also supports NumPy views with negative strides or read-only data.
        points = torch.from_numpy(coordinates.astype(np.float64, copy=True))
    else:
        if coordinates.is_complex() or coordinates.dtype == torch.bool:
            raise TypeError("coordinates must contain real numeric values.")
        points = coordinates.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(points).all():
        raise ValueError("coordinates must not contain NaN or Inf.")

    num_nodes = points.shape[0]
    if num_nodes < 2:
        return {
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_distance": torch.empty(0, dtype=torch.float64),
        }

    # Direct Euclidean computation avoids cancellation from the matrix-product
    # distance formula. Mask by index: distinct patches may share a center.
    distances = torch.cdist(points, points, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    if not torch.isfinite(distances).all():
        raise ValueError("Coordinate magnitudes are too large for finite distances.")
    distances.fill_diagonal_(float("inf"))
    neighbors = torch.argsort(distances, dim=1, stable=True)[:, :min(int(k), num_nodes - 1)]
    sources = torch.arange(num_nodes).unsqueeze(1).expand_as(neighbors)
    source, target = sources.reshape(-1), neighbors.reshape(-1)
    if max_distance is not None:
        keep = distances[source, target] <= max_distance
        source, target = source[keep], target[keep]

    edges = torch.stack((source, target))
    if bidirectional:
        edges = torch.cat((edges, edges.flip(0)), dim=1)
    # Integer pair keys deduplicate and impose deterministic source/target order.
    keys = torch.unique(edges[0] * num_nodes + edges[1], sorted=True)
    edge_index = torch.stack((keys // num_nodes, keys % num_nodes))
    edge_distance = distances[edge_index[0], edge_index[1]]
    return {"edge_index": edge_index, "edge_distance": edge_distance}
