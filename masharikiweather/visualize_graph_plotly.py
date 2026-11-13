import os
from typing import Any, Optional
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

def visualize_interactive_graph(batch: Any, output_dir: str = "plots", filename: str = "interactive_bipartite_graph.html") -> Optional[str]:
    """
    Creates an interactive Plotly visualization of the Heterogeneous Bipartite Graph.
    Uses physics-informed layouts (Kamada-Kawai) if true geographic coordinates (.pos) 
    are missing from the batch.
    """
    if nx is None:
        raise ImportError(
            "networkx is required for graph visualization. "
            "Please install it with: pip install networkx"
        )
    if go is None:
        raise ImportError(
            "plotly is required for interactive graph visualization. "
            "Please install it with: pip install plotly"
        )
    os.makedirs(output_dir, exist_ok=True)
    
    instance_G = nx.DiGraph()
    color_map = {'tahmo': '#ff7f0e', 'imerg': '#1f77b4', 'topography': '#2ca02c', 'chirps': '#9467bd'}
    
    # Extract edges from batch
    for src_type, relation, dst_type in batch.edge_types:
        if (src_type, relation, dst_type) not in batch.edge_indices():
            # Handle PyG HeteroData edge access safely
            if hasattr(batch, 'edge_index_dict') and (src_type, relation, dst_type) in batch.edge_index_dict:
                edge_index = batch.edge_index_dict[(src_type, relation, dst_type)]
            else:
                try:
                    edge_index = batch[src_type, relation, dst_type].edge_index
                except:
                    continue
        else:
            edge_index = batch[src_type, relation, dst_type].edge_index
            
        for i in range(edge_index.shape[1]):
            src_name = f"{src_type}_{edge_index[0, i].item()}"
            dst_name = f"{dst_type}_{edge_index[1, i].item()}"
            
            instance_G.add_node(src_name, type=src_type)
            instance_G.add_node(dst_name, type=dst_type)
            instance_G.add_edge(src_name, dst_name)

    # Subsample to keep the visualization fast (ego graph around a central grid or station)
    focal_nodes = [n for n in instance_G.nodes() if 'tahmo' in n]
    if focal_nodes:
        focal_node = focal_nodes[0]
        sub_G = nx.ego_graph(instance_G, focal_node, radius=2, undirected=True)
    else:
        sub_G = instance_G

    # Generate layout
    pos = nx.kamada_kawai_layout(sub_G)
    
    # Prepare Plotly Traces
    edge_x = []
    edge_y = []
    for edge in sub_G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = sub_G.nodes[node].get('type', 'unknown')
        node_text.append(f"{node}<br>Type: {ntype}")
        node_color.append(color_map.get(ntype, '#333'))
        # Stations are larger, grids are smaller
        node_size.append(15 if ntype == 'tahmo' else 8)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.split('_')[1] for n in sub_G.nodes()], # Show index as text
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2))

    # Build Figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'<br>Interactive Bipartite Sub-Graph (Ego: {focal_nodes[0] if focal_nodes else "None"})',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="MasharikiWeather - GNN Bipartite Topology",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
                    
    save_path = os.path.join(output_dir, filename)
    fig.write_html(save_path)
    print(f"Interactive graph saved to: {save_path}")
    return fig

if __name__ == "__main__":
    print("This module provides visualize_interactive_graph(batch).")
