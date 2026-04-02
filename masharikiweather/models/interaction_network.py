"""
Differentiable Physics Interaction Networks.

Implements the core MessagePassing framework utilized in GraphCast and ML LAMs.
Instead of purely relying on node embeddings, the network explicitly generates, updates, 
and aggregates Edge Attributes (Haversine Distance, Delta-Elevation, Topology) 
to simulate Earth System advection dynamics seamlessly over heterogeneous meshes.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class PhysicsInteractionBlock(MessagePassing):
    """
    Encode-Process-Decode PyG Block.
    Processes geographical priors natively on the edges, sidestepping the 
    VRAM explosion of dense Global Attention (GAT) while preventing the 
    physical "blindness" of GraphSAGE.
    """
    def __init__(self, channels: int, edge_dim: int):
        """
        Args:
            channels (int): Shared latent dimensionality (width) of the node encodings.
            edge_dim (int): Number of features in the edge matrix (E.g., 3 for [Dist, Weight, Elev]).
        """
        # aggr="mean" ensures numerical stability against highly varying localized node degrees
        super().__init__(aggr="mean")
        
        # 1. Edge MLP
        # Maps [sender_node, receiver_node, raw_edge_attrs] -> Updated Edge Embedding
        # Output stays in the generic 'channels' dimension
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * channels + edge_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )
        
        # 2. Node MLP
        # Maps [receiver_node_prior, aggregated_edge_messages] -> Updated Node Embedding
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Graph Forward Pass.
        
        Args:
            x (Tensor or Tuple[Tensor, Tensor]): Node embeddings.
            edge_index (LongTensor): [2, E] graph topology.
            edge_attr (Tensor): [E, edge_dim] static structural boundaries.
        """
        # PyG explicitly handles bipartite (Tuple) and unipartite (Tensor) inputs automatically
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages along every edge natively. O(E) complexity.
        """
        # Sanitize scalar edge attributes into 2D matrices
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # Vector Concatenation: Connect the Sender, Receiver, and Geo-Constraint
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute the hidden representation of the physical interaction
        return self.edge_mlp(edge_input)

    def update(self, aggr_out: torch.Tensor, x) -> torch.Tensor:
        """
        Skip-Connection Node Update.
        """
        # Bipartite detection (Grid -> Station) vs Unipartite (Station -> Station)
        if isinstance(x, tuple):
            x_target = x[1]
        else:
            x_target = x
            
        # Bind the incoming weather advection with the receiver's prior state
        node_input = torch.cat([x_target, aggr_out], dim=-1)
        
        # Residual connection avoids over-smoothing across deep message passing steps
        return x_target + self.node_mlp(node_input)
