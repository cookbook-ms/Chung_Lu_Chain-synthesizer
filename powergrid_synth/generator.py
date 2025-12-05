import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

# Import the components we built previously
from .preprocessing import Preprocessor
from .edge_creation import EdgeCreator
from .transformer_edges import TransformerConnector

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
                      keep_lcc: bool = True) -> nx.Graph:
        """
        Procedure CLCSTARS({d_xi}, {delta_xi}, {t[Xi, Xj]}) -> E
        
        Args:
            degrees_by_level: List of degree sequences, one for each voltage level.
            diameters_by_level: List of target diameters, one for each voltage level.
            transformer_degrees: Dictionary mapping level pairs (i, j) to a tuple of transformer degree lists.
            keep_lcc: If True, returns only the Largest Connected Component of the generated grid, removing isolated islands. Default: True

        Returns:
            A NetworkX graph representing the entire multi-level grid.
        """
        k = len(degrees_by_level)
        all_edges = set()
        
        # We need to track the global index offset for each level
        # to ensure node IDs don't clash between levels.
        # e.g., Level 0: 0-99, Level 1: 100-149
        level_offsets = [0] * k
        level_node_counts = [0] * k
        level_nodes_local_to_global = [] # List of mappings for each level

        current_global_offset = 0

        print(f"--- Starting Generation for {k} Voltage Levels ---")

        # =========================================================
        # Lines 3-7: Create each same-voltage subgraph
        # =========================================================
        for i in range(k):
            print(f"Generating Level {i}...")
            
            # 1. Run Algorithm 1 (SETUP)
            d_input = degrees_by_level[i]
            delta_input = diameters_by_level[i]
            
            d_prime, v, D, S = self.preprocessor.run_setup(d_input, delta_input)
            
            # 2. Run Algorithm 2 (CLC)
            local_edges = self.edge_creator.generate_edges(d_prime, v, D, S)
            
            # 3. Store Offset and Count
            n_nodes = len(d_prime)
            level_node_counts[i] = n_nodes
            level_offsets[i] = current_global_offset
            
            # 4. Convert local edges to global IDs and add to E
            for u, w in local_edges:
                global_u = u + current_global_offset
                global_w = w + current_global_offset
                all_edges.add((global_u, global_w))
                
            # Update offset for next level
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
                
                # Resize/Pad transformer degrees to match actual inflated node counts
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

                # Run Algorithm 3 (STARS)
                trans_edges = self.transformer_connector.generate_transformer_edges(t_i_j, t_j_i)
                
                # Convert to global IDs
                offset_i = level_offsets[i]
                offset_j = level_offsets[j]
                
                for u_local, v_local in trans_edges:
                    u_global = u_local + offset_i
                    v_global = v_local + offset_j
                    all_edges.add((u_global, v_global))

        # =========================================================
        # Build Final Graph
        # =========================================================
        G = nx.Graph()
        G.add_edges_from(all_edges)
        
        # Add metadata to nodes (ensure even isolated nodes are created with attrs)
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
                G = G.subgraph(largest_cc_nodes).copy()
                new_n = G.number_of_nodes()
                print(f"  -> Kept {new_n} nodes (removed {original_n - new_n} isolated/disconnected nodes).")
            else:
                print("Warning: Graph is empty, cannot filter for LCC.")
        
        return G