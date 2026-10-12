"""
Base Model Architectures utilizing the Physics Interaction Network blocks.

Demonstrates dynamic task scaling based on dataloader edge definitions.
No paths are hardcoded. If the dataset generates completely novel topological edges 
(e.g., forecasting target meshes), this network natively spawns matching processors 
and routes exactly as demanded by the dataloader outputs.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict, LayerNorm, GRU, MultiheadAttention
from torch_geometric.nn import HeteroConv
from typing import Dict, Tuple
from tsl.nn.blocks.encoders import RNN

class RetrospectiveTemporalCrossAttentionDecoder(torch.nn.Module):
    """
    Non-causal temporal cross-attention decoder for retrospective weather analysis.
    - Query (Q): Updated spatial GNN embedding [Nodes, Channels]
    - Key (K) & Value (V): Full local history sequence [Nodes, Time, Channels]
    """
    def __init__(self, channels: int, horizon_size: int, output_dim: int = 1, num_heads: int = 4):
        super().__init__()
        self.horizon_size = horizon_size
        self.output_dim = output_dim
        
        # Multihead attention (batch_first=True expects [Batch/Nodes, SeqLen, EmbedDim])
        self.cross_attn = MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = LayerNorm(channels)
        
        # Hurdle Model Heads (Occurrence + Intensity)
        self.occ_proj = Linear(channels, output_dim * horizon_size)
        self.int_proj = torch.nn.Sequential(
            Linear(channels, channels),
            torch.nn.SiLU(),
            Linear(channels, output_dim * horizon_size),
            torch.nn.Softplus()
        )

    def forward(self, z_spatial: torch.Tensor, h_temporal_seq: torch.Tensor):
        """
        Args:
            z_spatial: GNN spatial embeddings from HeteroConv [Nodes, Channels]
            h_temporal_seq: GRU temporal sequence states [Nodes, Time, Channels]
        Returns:
            logits: Occurrence logits [Nodes, Horizon, Output_Dim]
            intensity: Rainfall intensity [Nodes, Horizon, Output_Dim]
        """
        # 1. Expand spatial state to act as Query [Nodes, 1, Channels]
        query = z_spatial.unsqueeze(1)
        
        # 2. Assign Keys and Values from the full temporal history sequence
        key = h_temporal_seq
        value = h_temporal_seq
        
        # 3. Cross-Attention (No causal mask applied because task is retrospective)
        attn_out, _ = self.cross_attn(query=query, key=key, value=value, need_weights=False)
        
        # Residual connection and layer normalization
        z_decoded = self.norm(query + attn_out).squeeze(1) # [Nodes, Channels]
        
        # 4. Project to Two-Headed Hurdle outputs
        n_nodes = z_decoded.shape[0]
        logits = self.occ_proj(z_decoded).reshape(n_nodes, self.horizon_size, self.output_dim)
        intensity = self.int_proj(z_decoded).reshape(n_nodes, self.horizon_size, self.output_dim)
        
        return logits, intensity


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
        # Local Temporal Encoders (Per Node Type)
        # Operates on [Nodes, Time, Features] to capture sequential storm patterns
        # We use TSL's RNN block natively
        self.temporal_encoders = ModuleDict({
            node_type: RNN(
                input_size=in_channels, 
                hidden_size=hidden_channels, 
                return_only_last_state=False,
                n_layers=2,
                dropout=0.1,
                cell='gru'
            )
            for node_type, in_channels in in_channels_dict.items()
        })
        
        # Fallback for static nodes (like topography) that lack a time dimension
        self.static_encoders = ModuleDict({
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
        self.decoder = RetrospectiveTemporalCrossAttentionDecoder(
            channels=hidden_channels, 
            horizon_size=self.horizon_size, 
            output_dim=self.tahmo_params
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict, edge_attr_dict: Dict):
        """
        Task-agnostic graph projection sequence.
        """
        # A. Encode
        z_dict = {}
        h_seq_dict = {}
        for node_type, x in x_dict.items():
            z = None
            if x.dim() == 3:
                # [Nodes, Time, Features] -> [Nodes, Time, hidden_channels]
                if node_type in self.temporal_encoders:
                    h_seq = self.temporal_encoders[node_type](x)
                    h_seq_dict[node_type] = h_seq
                    z = h_seq[:, -1, :] # Last state for GNN
            else:
                # Fallback for static nodes without a time dimension (e.g., topography)
                if node_type in self.static_encoders:
                    z = self.static_encoders[node_type](x)

            if z is not None and node_type in self.norms:
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
        if 'tahmo' in z_dict and 'tahmo' in h_seq_dict:
            z_tahmo = F.silu(z_dict['tahmo'])
            h_seq_tahmo = h_seq_dict['tahmo']
            
            logits, intensity = self.decoder(z_tahmo, h_seq_tahmo)
            
            return logits, intensity
        
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
        if hasattr(data, 'constraint_time_index') and data.constraint_time_index is not None:
            # Flexible temporal mappings
            num_days = int(data.constraint_time_index.max().item()) + 1
            daily_preds = []
            for d in range(num_days):
                mask = (data.constraint_time_index == d)
                daily_preds.append(pred_hourly[:, mask].sum(dim=1))
            daily_pred = torch.stack(daily_preds, dim=1)
        else:
            # Assume naive 24-hr multiples if constraint_time_index isn't provided
            n_nodes, horizon, channels = pred_hourly.shape
            daily_pred = pred_hourly.view(n_nodes, -1, 24, channels).sum(dim=2)
            
        loss = tweedie(daily_pred, data.mass_constraint, mask=getattr(data, 'mass_mask', None))
        return loss * lambda_mass


class BaselinePrecipAnomalyDetector(torch.nn.Module):
    """
    A simple baseline Spatiotemporal GNN for Precipitation Anomaly Detection.
    Uses basic GRU encoding (last state), standard HeteroConv spatial processing,
    and a simple Linear/MLP decoder predicting raw precipitation directly via Softplus.
    Designed for straightforward MAE/MSE demonstration.
    """
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        edge_dim_dict: Dict[Tuple[str, str, str], int],
        hidden_channels: int,
        window_size: int,
        horizon_size: int,
        tahmo_params: int,
        static_channels_dict: Dict[str, int] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.tahmo_params = tahmo_params

        # 1. Node Encoder (Basic 1-layer GRU, keeping only the final state)
        self.temporal_encoders = ModuleDict({
            node_type: RNN(
                input_size=in_channels, 
                hidden_size=hidden_channels, 
                return_only_last_state=True,
                n_layers=1,
                dropout=0.0,
                cell='gru'
            )
            for node_type, in_channels in in_channels_dict.items()
        })
        
        self.static_encoders = ModuleDict({
            node_type: Linear(in_channels, hidden_channels)
            for node_type, in_channels in in_channels_dict.items()
        })
        
        # Terrain Encoders and Fusion
        self.terrain_encoders = ModuleDict()
        self.fusion = ModuleDict()
        
        if static_channels_dict is not None:
            for node_type, static_channels in static_channels_dict.items():
                self.terrain_encoders[node_type] = torch.nn.Sequential(
                    Linear(static_channels, 16),
                    torch.nn.ReLU()
                )
                if node_type in self.temporal_encoders:
                    # We will fuse the 16-dim terrain embedding with the hidden_channels dynamic embedding
                    self.fusion[node_type] = Linear(hidden_channels + 16, hidden_channels)

        self.norms = ModuleDict({
            node_type: LayerNorm(hidden_channels)
            for node_type in in_channels_dict.keys()
        })

        # 2. Spatial Processor (Simple HeteroConv with SAGEConv)
        from .interaction_network import BaselineInteractionBlock
        self.processor = BaselineInteractionBlock(
            channels=hidden_channels,
            edge_types=list(edge_dim_dict.keys())
        )

        # 3. Decoder (Simple MLP to raw precipitation)
        self.decoder = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, self.horizon_size * self.tahmo_params),
            torch.nn.Softplus() # Ensure precipitation is non-negative
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict, edge_attr_dict: Dict = None, static_u_dict: Dict = None):
        """
        Baseline forward pass returning a single intensity tensor.
        """
        if static_u_dict is None:
            static_u_dict = {}
            
        # A. Encode
        z_dict = {}
        for node_type, x in x_dict.items():
            z = None
            if x.dim() == 3:
                if node_type in self.temporal_encoders:
                    z = self.temporal_encoders[node_type](x)
                    
                    # Fuse static terrain features if available for this temporal node
                    if node_type in self.terrain_encoders and node_type in static_u_dict:
                        terrain_h = self.terrain_encoders[node_type](static_u_dict[node_type])
                        z_fused = torch.cat([z, terrain_h], dim=-1)
                        z = F.relu(self.fusion[node_type](z_fused))
            else:
                if node_type in self.static_encoders:
                    z = self.static_encoders[node_type](x)

            if z is not None and node_type in self.norms:
                z = self.norms[node_type](z)
                z_dict[node_type] = F.relu(z)

        # B. Process
        z_dict = self.processor(
            x_dict=z_dict, 
            edge_index_dict=edge_index_dict
        )

        # C. Decode
        if 'tahmo' in z_dict:
            z_tahmo = F.relu(z_dict['tahmo'])
            n_nodes = z_tahmo.shape[0]
            
            # Predict intensity directly [Nodes, Horizon, Params]
            intensity = self.decoder(z_tahmo).reshape(n_nodes, self.horizon_size, self.tahmo_params)
            return intensity
        
        return z_dict
