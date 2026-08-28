"""
Base Model Architectures utilizing the Physics Interaction Network blocks.

Demonstrates dynamic task scaling based on dataloader edge definitions.
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
        horizon_size: int,
        tahmo_params: int
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.tahmo_params = tahmo_params

        # -----------------------------------------------------
        # 1. The Encoders
        # -----------------------------------------------------
        # Independent linear layers map disparate data sources (e.g IMERG, ERA5, TAHMO)
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
        # 2. The Processor 
        # -----------------------------------------------------
        # spawns PhysicsInteractionBlocks for EVERY edge detected 
        # by the dataloader.
        # task agnostic
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
        self.decoder = Linear(hidden_channels, self.tahmo_params * self.horizon_size)

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

            # D. Unflatten back to physical reality [Nodes, Horizon, Channels]
            return reconstructed_flat.reshape(-1, self.horizon_size, self.tahmo_params)
        
        # Fallback if predicting diverse target grids instead of specific stations
        return z_dict

    def mass_conservation_loss(self, pred_hourly: torch.Tensor, data, lambda_mass: float = 1.0, p: float = 1.5):
        """
        Optional Mass Conservation Constraint.
        aggregates the decoded hourly predictions and penalizes them against 
        low-frequency physically-conserved outputs (CHIRPS/TAMSAT daily arrays).

        relies on `losses.TweedieDevianceLoss` to prevent zero-bounded MSE crashes.
        """
        if lambda_mass <= 0.0 or not hasattr(data, 'mass_constraint') or data.mass_constraint is None:
            return 0.0
            
        from .losses import TweedieDevianceLoss
        tweedie = TweedieDevianceLoss(p=p)
        
        # Determine continuous daily clusters from 1D boolean array or assume continuous 24h strides
        if hasattr(data, 'day_index') and data.day_index is not None:
            # Flexible temporal mappings
            num_days = int(data.day_index.max().item()) + 1
            daily_preds = []
            for d in range(num_days):
                mask = (data.day_index == d)
                daily_preds.append(pred_hourly[:, mask].sum(dim=1))
            daily_pred = torch.stack(daily_preds, dim=1)
        else:
            # Assume naive 24-hr multiples if day_index isn't provided
            n_nodes, horizon, channels = pred_hourly.shape
            daily_pred = pred_hourly.view(n_nodes, -1, 24, channels).sum(dim=2)
            
        loss = tweedie(daily_pred, data.mass_constraint, mask=getattr(data, 'mass_mask', None))
        return loss * lambda_mass
