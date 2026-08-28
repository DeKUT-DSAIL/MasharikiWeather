"""
Differentiable Physics Interaction Networks.

Implements the core MessagePassing framework.
Instead of purely relying on node embeddings, the network explicitly generates, updates, 
and aggregates Edge Attributes (Haversine Distance, Delta-Elevation, Topology) 
to simulate Earth System advection dynamics over heterogeneous meshes.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import softmax

class PhysicsInteractionBlock(MessagePassing):
    """
    Encode-Process-Decode PyG Block.
    Processes geographical priors natively on the edges, avoiding the
    VRAM explosion of dense Global Attention (GAT) while preventing the
    physical "blindness" of GraphSAGE.
    """
    def __init__(self, channels: int, edge_dim: int):
        """
        Args:
            channels (int): Shared latent dimensionality (width) of the node encodings.
            edge_dim (int): Number of features in the edge matrix (E.g., 3 for [Dist, Weight, Elev]).
        """
        # aggr="add" is used because we apply data-dependent spatial attention (softmax)
        # to properly scale Grid -> Station message passing based on topographic relevance.
        super().__init__(aggr="add", node_dim=0)
        
        self.channels = channels
        self.heads = 4
        assert channels % self.heads == 0, "channels must be divisible by heads"
        self.head_dim = channels // self.heads
        
        # 1. Edge MLP
        # Maps [sender_node, receiver_node, raw_edge_attrs] -> Updated Edge Embedding
        # Output stays in the generic 'channels' dimension
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * channels + edge_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )
        
        # Attention projection to compute scores from the physical edge embedding
        self.attn_proj = nn.Linear(channels, self.heads)
        
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

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor:
        """
        Constructs messages along every edge natively and applies spatial attention.
        """
        # Sanitize scalar edge attributes into 2D matrices
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # Vector Concatenation: Connect the Sender, Receiver, and Geo-Constraint
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute the hidden representation of the physical interaction
        edge_emb = self.edge_mlp(edge_input)
        
        # Physics-informed spatial attention weights
        alpha = self.attn_proj(edge_emb)
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Apply multi-head attention weighting
        edge_emb = edge_emb.view(-1, self.heads, self.head_dim)
        alpha = alpha.view(-1, self.heads, 1)
        msg = (edge_emb * alpha).view(-1, self.channels)
        
        return msg

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
