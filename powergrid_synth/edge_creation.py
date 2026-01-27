r"""
This module connects the edges within each same-voltage level subgraph. 
"""

import numpy as np
import random
from typing import List, Set, Tuple, Dict

class EdgeCreator:
    """
    Generative model for subgraph (Edge Creation).
    """

    def generate_edges(self, d_prime: np.ndarray, v: np.ndarray, D: Set[int], S: Set[int]) -> List[Tuple[int, int]]:
        """
        Procedure CLC(d', v, D, S) -> E
        
        Args:
            d_prime: Updated degree sequence (numpy array).
            v: Vertex-box list (numpy array where v[i] is box ID).
            D: Set of diameter path vertices.
            S: Set of subdiameter path vertices.
            
        Returns:
            List of edges (tuples of node indices).
        """
        edges = set()

        # --- Lines 3-7: Make Diameter Path ---
        # The paper iterates k=1..|D|-1 and finds nodes with box k and k+1.
        # Since our boxes might be 0-indexed or non-contiguous (for S), 
        # a robust equivalent is to sort the path nodes by their box ID 
        # and connect adjacent ones.
        
        d_path_nodes = sorted(list(D), key=lambda node_idx: v[node_idx])
        
        # Connect node in Box k to node in Box k+1
        for k in range(len(d_path_nodes) - 1):
            u, w = d_path_nodes[k], d_path_nodes[k+1]
            # Ensure we are connecting nodes in adjacent boxes (just a sanity check)
            # Logic: The path is defined by the spatial sequence of boxes.
            edges.add(tuple(sorted((u, w))))

        # --- Lines 8-12: Make Subdiameter Path ---
        s_path_nodes = sorted(list(S), key=lambda node_idx: v[node_idx])
        
        for k in range(len(s_path_nodes) - 1):
            u, w = s_path_nodes[k], s_path_nodes[k+1]
            edges.add(tuple(sorted((u, w))))

        # --- Lines 13-22: Chung-Lu graph in each box ---
        # "For k = 1...max(v)" -> Iterate over all active boxes
        unique_boxes = np.unique(v)
        # Filter out -1 (unassigned) if any exists (though Algorithm 1 should fill them)
        unique_boxes = unique_boxes[unique_boxes != -1]

        for box_id in unique_boxes:
            # Line 15: B_k <- {j : v_j = k} (All vertices in box k)
            B_k = np.where(v == box_id)[0]
            
            if len(B_k) < 2:
                continue

            # Line 16: m_k <- round(0.5 * sum(d_i for i in B_k))
            degrees_in_box = d_prime[B_k]
            sum_degrees = np.sum(degrees_in_box)
            m_k = int(round(0.5 * sum_degrees))
            
            if m_k <= 0:
                continue

            # Lines 17-21: Generate m_k edges probabilistically
            # We use random.choices for weighted sampling with replacement.
            # Weight for node i is d_i.
            
            # Select 2 * m_k nodes (to form m_k pairs)
            # This is equivalent to "Select i proportional to d_i, Select j proportional to d_j"
            # repeated m_k times.
            nodes_selected = random.choices(B_k, weights=degrees_in_box, k=2 * m_k)
            
            # Pair them up
            for idx in range(0, len(nodes_selected), 2):
                i = nodes_selected[idx]
                j = nodes_selected[idx+1]
                
                # Line 20: E <- E U {i, j} (Discard loops)
                if i != j:
                    # Sort to ensure (u, v) is same as (v, u) in set
                    edges.add(tuple(sorted((i, j))))

        return list(edges)