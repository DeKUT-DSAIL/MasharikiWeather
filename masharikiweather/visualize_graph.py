import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_weather_graph(batch, max_nodes_per_type=15, output_dir="plots"):
    """
    Generates and saves two visualizations: The Graph Schema and a Subsampled Instance.
    
    Args:
        batch: The PyTorch Geometric HeteroData batch object.
        max_nodes_per_type (int): Maximum number of nodes to render per modality.
        output_dir (str): Directory where the high-resolution plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. PLOT THE SCHEMA (Meta-Graph)
    # ==========================================
    schema_G = nx.DiGraph()
    
    for src, rel, dst in batch.edge_types:
        schema_G.add_edge(src, dst, label=rel)
        
    plt.figure(figsize=(8, 6))
    
    # Use a planar layout so the nodes separate cleanly
    pos_schema = nx.spring_layout(schema_G, seed=42, k=2.0)
    
    nx.draw(schema_G, pos_schema, with_labels=True, node_color='lightblue', 
            node_size=4000, font_size=12, font_weight='bold', arrows=True, 
            edge_color='gray', width=2, arrowsize=20)
    
    edge_labels = nx.get_edge_attributes(schema_G, 'label')
    nx.draw_networkx_edge_labels(schema_G, pos_schema, edge_labels=edge_labels, 
                                 font_color='red', font_size=10)
    
    plt.title("1. Heterogeneous Graph Schema (The Architecture)", fontsize=14, fontweight='bold')
    plt.margins(0.2)
    
    # Save the schema plot before showing
    schema_save_path = os.path.join(output_dir, "heterogeneous_graph_schema.png")
    plt.savefig(schema_save_path, dpi=300, bbox_inches='tight')
    print(f"Schema graph saved to: {schema_save_path}")
    
    plt.show()

    # ==========================================
    # 2. PLOT THE INSTANCE (Ego-Graph Sub-Sampling)
    # ==========================================
    instance_G = nx.DiGraph()
    
    # Color coding dictionary for different modalities
    color_map = {'tahmo': '#ff7f0e', 'imerg': '#1f77b4', 'topography': '#2ca02c', 'era5': '#9467bd'}
    
    # Step A: Load ALL edges into NetworkX (NetworkX can handle a single batch easily)
    for src_type, relation, dst_type in batch.edge_types:
        edge_index = batch[src_type, relation, dst_type].edge_index
        
        for i in range(edge_index.shape[1]):
            src_name = f"{src_type}_{edge_index[0, i].item()}"
            dst_name = f"{dst_type}_{edge_index[1, i].item()}"
            
            instance_G.add_node(src_name, type=src_type)
            instance_G.add_node(dst_name, type=dst_type)
            instance_G.add_edge(src_name, dst_name)

    # Step B: Sub-sample geographically around a focal point (e.g., the first TAHMO station)
    focal_node = "tahmo_0"
    
    if focal_node in instance_G:
        # Extract a 2-hop radius around the focal station (Macro grids -> Micro grids -> Station)
        sub_G = nx.ego_graph(instance_G, focal_node, radius=2, undirected=True)
    else:
        # Fallback if no TAHMO nodes exist in this batch
        sub_G = instance_G 

    # Assign colors based on the node type
    node_colors = [color_map.get(sub_G.nodes[n].get('type', ''), 'gray') for n in sub_G.nodes()]
    
    plt.figure(figsize=(12, 8))
    
    # Kamada-Kawai layout is much better for physics/grid meshes than Spring layout
    pos_instance = nx.kamada_kawai_layout(sub_G)
    
    nx.draw(sub_G, pos_instance, node_color=node_colors, with_labels=True, 
            node_size=600, font_size=8, edge_color='lightgray', arrows=True, arrowsize=10)
    
    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=12) 
                       for k, v in color_map.items() if k in batch.node_types]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.title(f"2. Physical Ego-Graph (2-Hop Radius around {focal_node})", fontsize=14, fontweight='bold')
    
    # Save the instance plot before showing
    instance_save_path = os.path.join(output_dir, "subsampled_physical_graph.png")
    plt.savefig(instance_save_path, dpi=300, bbox_inches='tight')
    print(f"Instance graph saved to: {instance_save_path}")
    
    plt.show()