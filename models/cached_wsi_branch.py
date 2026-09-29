"""Training execution from fixed raw patch features; validated GAT mathematics."""
from mcminet.config import default

from collections.abc import Sequence
import torch
from torch import nn
from mcminet.data.wsi_feature_cache import validate_tensors
from mcminet.data.wsi_graph_builder import build_spatial_graph
from mcminet.models.wsi_gat_encoder import WSIGATEncoder


class CachedWSIBranch(nn.Module):
    def __init__(self, gat_encoder=None, *, graph_config=None):
        super().__init__()
        self.graph_config = default("graph") if graph_config is None else dict(graph_config)
        self.gat_encoder=gat_encoder if gat_encoder is not None else WSIGATEncoder(input_dim=default("WSI.node_feature_dim"))

    def forward(self,features_list,coordinates_list,return_details=False,return_attention=False):
        for values in (features_list,coordinates_list):
            if isinstance(values,(torch.Tensor,str,bytes)) or not isinstance(values,Sequence):raise TypeError('Patient tensor sequences required')
        if not features_list or len(features_list)!=len(coordinates_list):raise ValueError('Nonempty matching patient sequences required')
        counts=[];edges=[];distances=[];offset=0
        for features,coordinates in zip(features_list,coordinates_list):
            validate_tensors(features,coordinates)
            graph=build_spatial_graph(coordinates,**self.graph_config)
            counts.append(len(features));edges.append(graph['edge_index']+offset);distances.append(graph['edge_distance']);offset+=len(features)
        device=next(self.gat_encoder.parameters()).device
        # Normal detached tensors; no patch encoder is owned or invoked here.
        with torch.inference_mode(False):
            x=torch.cat([f.detach().clone() for f in features_list]).to(device)
        edge_index=torch.cat(edges,dim=1).to(device);edge_distance=torch.cat(distances)
        batch=torch.repeat_interleave(torch.arange(len(counts),device=device),torch.tensor(counts,device=device))
        previous=self.gat_encoder.return_attention
        try:
            self.gat_encoder.return_attention=return_attention
            output=self.gat_encoder(x,edge_index,batch)
        finally:self.gat_encoder.return_attention=previous
        if not return_details and not return_attention:return output['embedding']
        return dict(embedding=output['embedding'],node_features=x,edge_index=edge_index,edge_distance=edge_distance,
                    batch=batch,patch_counts=torch.tensor(counts,device=device),gat_output=output,attention=output['attention'])
