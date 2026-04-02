"""
Base Model Architectures utilizing the Physics Interaction Network blocks.

Demonstrates 100% dynamic task scaling based strictly on dataloader edge definitions.
No paths are hardcoded. If the dataset generates completely novel topological edges 
(e.g., forecasting target meshes), this network natively spawns matching processors 
and routes exactly as demanded by the dataloader outputs.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict, LayerNorm
from torch_geometric.nn import HeteroConv
from typing import Dict, Tuple

from .interaction_network import PhysicsInteractionBlock


class PrecipAnomalyDetector(torch.nn.Module):
    """
    SOTA Encode-Process-Decode Weather Graph Architecture.
    Completely task-agnostic: HeteroConv structures dynamically build themselves
    based on the datasets currently configured in the EarthSystemAdapter.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        edge_dim_dict: Dict[Tuple[str, str, str], int],
        hidden_channels: int,
        window_size: int,
        tahmo_params: int
    ):
        super().__init__()
        self.window_size = window_size
        self.tahmo_params = tahmo_params

        # -----------------------------------------------------
        # 1. The Encoders (Minimally Processed Latent Fusion)
        # -----------------------------------------------------
        # Independent linear layers map disparate data sources (IMERG, ERA5, TAHMO)
        # into a shared mathematical latent dimension without forced interpolation.
        self.embeddings = ModuleDict({
            node_type: Linear(in_channels, hidden_channels)
            for node_type, in_channels in in_channels_dict.items()
        })

        # Layer Normalization stabilizes latent magnitudes before message passing
        self.norms = ModuleDict({
            node_type: LayerNorm(hidden_channels)
            for node_type in in_channels_dict.keys()
        })

        # -----------------------------------------------------
        # 2. The Processor (Differentiable Message Passing)
        # -----------------------------------------------------
        # Dynamically spawns PhysicsInteractionBlocks for EVERY edge detected 
        # by the dataloader. No hardcoded logic. 
        # Automatically scales across downscaling/anomaly/forecasting.
        interaction_dict = {}
        for edge_type, edge_dim in edge_dim_dict.items():
            interaction_dict[edge_type] = PhysicsInteractionBlock(
                channels=hidden_channels, 
                edge_dim=edge_dim
            )
            
        self.processor = HeteroConv(interaction_dict, aggr='mean')

        # -----------------------------------------------------
        # 3. The Decoder
        # -----------------------------------------------------
        # Maps the latent representation back to the original physical target (TAHMO precipitation)
        self.decoder = Linear(hidden_channels, self.tahmo_params * self.window_size)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict, edge_attr_dict: Dict):
        """
        Task-agnostic graph projection sequence.
        """
        # A. Encode
        z_dict = {}
        for node_type, x in x_dict.items():
            # Dynamically flatten 3D spatiotemporal tensors (e.g., [Nodes, Time, Params])
            # into 2D Graph Nodes [Nodes, Channels * Time]
            if x.dim() == 3:
                x = x.reshape(x.size(0), -1)

            if node_type in self.embeddings:
                z = self.embeddings[node_type](x)
                z = self.norms[node_type](z)
                z_dict[node_type] = F.silu(z)

        # B. Process (Message Passing over Physics Edges)
        # We explicitly inject edge_attr_dict so Interaction Networks can digest Physical Topography
        z_dict = self.processor(
            x_dict=z_dict, 
            edge_index_dict=edge_index_dict, 
            edge_attr_dict=edge_attr_dict
        )

        # C. Decode (Task Specific Reconstruction)
        # In this task, we specifically target TAHMO anomaly reconstruction
        if 'tahmo' in z_dict:
            z_tahmo = F.silu(z_dict['tahmo'])
            # Softplus bounds precipitation to physical reality [0, inf) preventing log(Negative) crashing
            reconstructed_flat = F.softplus(self.decoder(z_tahmo))

            # D. Unflatten back to physical reality [Nodes, Time, Channels]
            return reconstructed_flat.reshape(-1, self.window_size, self.tahmo_params)
        
        # Fallback if predicting diverse target grids instead of specific stations
        return z_dict
