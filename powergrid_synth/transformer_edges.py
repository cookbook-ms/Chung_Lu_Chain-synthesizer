import numpy as np
import random
from typing import List, Set, Tuple, Dict

class TransformerConnector:
    """
    Implements Algorithm 3: Insert transformer edges between subgraphs of voltage X and Y.
    """

    def generate_transformer_edges(self, t_xy: List[int], t_yx: List[int]) -> List[Tuple[int, int]]:
        """
        Procedure STARS(t[X,Y], t[Y,X]) -> E
        
        Args:
            t_xy: List of transformer degrees for nodes in subgraph X.
                  (i.e., how many connections node i in X has to Y)
            t_yx: List of transformer degrees for nodes in subgraph Y.
                  (i.e., how many connections node j in Y has to X)
            
        Returns:
            List of edges (u, v) where u is index in X, v is index in Y.
        """
        # Line 2: Initialize
        edges = set()
        L_x = [] # Leftover bins for X
        L_y = [] # Leftover bins for Y

        # Line 3-4: Categorize vertices
        # We store them as lists to allow efficient random sampling/shuffling
        # Indices are 0-based relative to the input arrays
        
        # Degree 1 vertices (Open) - "Pools"
        I_x_o = [i for i, d in enumerate(t_xy) if d == 1]
        I_y_o = [i for i, d in enumerate(t_yx) if d == 1]
        
        # Degree >= 2 vertices (Centers)
        I_x_c = [i for i, d in enumerate(t_xy) if d >= 2]
        I_y_c = [i for i, d in enumerate(t_yx) if d >= 2]
        
        # Shuffle lists to simulate "Randomly select" efficiently when we pop()
        random.shuffle(I_x_o)
        random.shuffle(I_y_o)
        random.shuffle(I_x_c)
        random.shuffle(I_y_c)

        # --- Lines 7-23 & 25: Run k-STARS(X, Y) ---
        # Create k-stars centered in X with leaves in Y
        for i in I_x_c:
            req_degree = t_xy[i]
            
            # Line 11: Check if enough leaf vertices in Y
            if len(I_y_o) >= req_degree:
                # Line 12-16: Connect i to multiple j's
                for _ in range(req_degree):
                    j = I_y_o.pop() # Remove leaf vertex from pool
                    edges.add((i, j))
                # i is consumed (removed from I_x_c loop effectively)
            else:
                # Line 18-20: Not enough leaves, put i in leftover
                L_x.append(i)

        # --- Lines 7-23 & 26: Run k-STARS(Y, X) ---
        # Create k-stars centered in Y with leaves in X
        for i in I_y_c:
            req_degree = t_yx[i]
            
            # Check if enough leaf vertices in X
            if len(I_x_o) >= req_degree:
                for _ in range(req_degree):
                    j = I_x_o.pop() # Remove leaf
                    # Edge is (u, v) -> (X-index, Y-index)
                    # Here j is from X, i is from Y
                    edges.add((j, i)) 
            else:
                L_y.append(i)

        # --- Lines 27-32: Insert 1-stars (edges) on remaining degree 1 vertices ---
        # While both pools have elements
        while I_x_o and I_y_o:
            i = I_x_o.pop()
            j = I_y_o.pop()
            edges.add((i, j))

        # --- Line 33: Update Leftovers ---
        # Add remaining unmatched degree-1 nodes to leftovers
        L_x.extend(I_x_o)
        L_y.extend(I_y_o)

        # --- Lines 34-42: Bipartite Chung-Lu on leftovers ---
        if L_x:
            # Line 36: Total edges to insert (m)
            # Sum of desired degrees for nodes in L_x
            m = sum(t_xy[i] for i in L_x)
            
            if m > 0 and L_y:
                # Prepare weights for weighted sampling
                weights_x = [t_xy[i] for i in L_x]
                weights_y = [t_yx[j] for j in L_y]
                
                # Check for zero sum weights (safety against empty or all-zero degree pools)
                if sum(weights_y) > 0 and sum(weights_x) > 0:
                    # Line 37: Loop m times
                    # Efficient sampling: random.choices returns list of size k (m)
                    # We select 'm' pairs roughly proportional to their degrees
                    chosen_xs = random.choices(L_x, weights=weights_x, k=m)
                    chosen_ys = random.choices(L_y, weights=weights_y, k=m)
                    
                    for i, j in zip(chosen_xs, chosen_ys):
                        # Line 40: Discard duplicate edges
                        if (i, j) not in edges:
                            edges.add((i, j))

        return list(edges)