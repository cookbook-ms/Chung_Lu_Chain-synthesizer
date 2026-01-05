import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

# Import the components we built previously
from .preprocessing import Preprocessor
from .edge_creation import EdgeCreator
from .transformer_edges import TransformerConnector

class PowerGridGraph(nx.Graph):
    """
    Custom NetworkX Graph for Power Grids.
    Extends nx.Graph to support subgraph extraction by voltage level.
    """
    def level(self, voltage_level: int) -> nx.Graph:
        """
        Returns a subgraph view containing only nodes at the specified voltage level.
        
        Args:
            voltage_level (int): The voltage level index (e.g., 0, 1).
            
        Returns:
            nx.Graph: A subgraph view of the grid.
        """
        nodes = [n for n, d in self.nodes(data=True) if d.get('voltage_level') == voltage_level]
        return self.subgraph(nodes)

class PowerGridGenerator:
    """
    Implements Algorithm 4: Generative model for entire power grid graph on k voltage levels.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        self.preprocessor = Preprocessor()
        self.edge_creator = EdgeCreator()
        self.transformer_connector = TransformerConnector()

    def generate_grid(self, 
                      degrees_by_level: List[List[int]], 
                      diameters_by_level: List[int], 
                      transformer_degrees: Dict[Tuple[int, int], Tuple[List[int], List[int]]],
                      keep_lcc: bool = False) -> PowerGridGraph:
        """
        Procedure CLCSTARS({d_xi}, {delta_xi}, {t[Xi, Xj]}) -> E
        
        Args:
            degrees_by_level: List of degree sequences, one for each voltage level.
            diameters_by_level: List of target diameters, one for each voltage level.
            transformer_degrees: Dictionary mapping level pairs (i, j) to a tuple of transformer degree lists.
            keep_lcc: If True, returns only the Largest Connected Component of the generated grid.

        Returns:
            A PowerGridGraph representing the entire multi-level grid.
        """
        k = len(degrees_by_level)
        all_edges = set()
        
        level_offsets = [0] * k
        level_node_counts = [0] * k
        
        current_global_offset = 0

        print(f"--- Starting Generation for {k} Voltage Levels ---")

        # =========================================================
        # Lines 3-7: Create each same-voltage subgraph
        # =========================================================
        for i in range(k):
            print(f"Generating Level {i}...")
            
            d_input = degrees_by_level[i]
            delta_input = diameters_by_level[i]
            
            d_prime, v, D, S = self.preprocessor.run_setup(d_input, delta_input)
            local_edges = self.edge_creator.generate_edges(d_prime, v, D, S)
            
            n_nodes = len(d_prime)
            level_node_counts[i] = n_nodes
            level_offsets[i] = current_global_offset
            
            for u, w in local_edges:
                global_u = u + current_global_offset
                global_w = w + current_global_offset
                all_edges.add((global_u, global_w))
                
            current_global_offset += n_nodes
            print(f"  -> Level {i} Complete. Nodes: {n_nodes}, Edges: {len(local_edges)}")

        # =========================================================
        # Lines 8-13: Insert transformer edges between levels
        # =========================================================
        print("Generating Transformer Connections...")
        for i in range(k):
            for j in range(i + 1, k):
                
                if (i, j) not in transformer_degrees:
                    continue
                
                print(f"  -> Connecting Level {i} <-> Level {j}")
                
                t_i_j_input, t_j_i_input = transformer_degrees[(i, j)]
                
                actual_n_i = level_node_counts[i]
                t_i_j = list(t_i_j_input)
                if len(t_i_j) < actual_n_i:
                    t_i_j.extend([0] * (actual_n_i - len(t_i_j)))
                elif len(t_i_j) > actual_n_i:
                    t_i_j = t_i_j[:actual_n_i]
                
                actual_n_j = level_node_counts[j]
                t_j_i = list(t_j_i_input)
                if len(t_j_i) < actual_n_j:
                    t_j_i.extend([0] * (actual_n_j - len(t_j_i)))
                elif len(t_j_i) > actual_n_j:
                    t_j_i = t_j_i[:actual_n_j]

                trans_edges = self.transformer_connector.generate_transformer_edges(t_i_j, t_j_i)
                
                offset_i = level_offsets[i]
                offset_j = level_offsets[j]
                
                for u_local, v_local in trans_edges:
                    u_global = u_local + offset_i
                    v_global = v_local + offset_j
                    all_edges.add((u_global, v_global))

        # =========================================================
        # Build Final Graph
        # =========================================================
        G = PowerGridGraph() # Use Custom Class directly
        G.add_edges_from(all_edges)
        
        for i in range(k):
            start = level_offsets[i]
            end = start + level_node_counts[i]
            for node_id in range(start, end):
                G.add_node(node_id, voltage_level=i)

        # =========================================================
        # Post-Processing: Largest Connected Component (Optional)
        # =========================================================
        if keep_lcc:
            if G.number_of_nodes() > 0:
                print("Filtering for Largest Connected Component (LCC)...")
                original_n = G.number_of_nodes()
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                
                # We need to preserve the PowerGridGraph class type
                sub_view = G.subgraph(largest_cc_nodes)
                G_new = PowerGridGraph()
                G_new.add_nodes_from(sub_view.nodes(data=True))
                G_new.add_edges_from(sub_view.edges(data=True))
                # Update graph-level attributes
                G_new.graph.update(sub_view.graph)
                G = G_new
                
                new_n = G.number_of_nodes()
                print(f"  -> Kept {new_n} nodes (removed {original_n - new_n} isolated nodes).")
            else:
                print("Warning: Graph is empty, cannot filter for LCC.")
        
        return G