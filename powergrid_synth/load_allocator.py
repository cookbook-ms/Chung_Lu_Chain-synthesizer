import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Optional
from .reference_data import get_reference_stats

class LoadAllocator:
    """
    Assigns active power loads (PL) to load buses in the grid.
    Ported and adapted from 'sg_load.m' (SynGrid).
    """

    def __init__(self, graph: nx.Graph, ref_sys_id: int = 1):
        """
        Args:
            graph: The networkx graph with 'bus_type' and 'pg_max' attributes.
            ref_sys_id: ID of reference system statistics to use (1, 2, or 3).
                        Use 0 for heuristic default.
        """
        self.graph = graph
        self.ref_sys_id = ref_sys_id
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        
        # Identify Load Buses (Bus Type = 2)
        # If attribute missing, defaults to empty list
        self.load_buses = [n for n, d in graph.nodes(data=True) if d.get('bus_type') == 'Load']
        self.n_load = len(self.load_buses)

    def _get_tab_2d_load(self) -> np.ndarray:
        """
        Returns the Tab_2D_load from the selected reference system.
        """
        # Heuristic fallback (Option 0)
        if self.ref_sys_id == 0:
            # Create a diagonal-heavy matrix similar to the capacity heuristic
            matrix = np.zeros((14, 14))
            for r in range(14):
                for c in range(14):
                    dist = abs(r - c)
                    matrix[r, c] = np.exp(-0.5 * dist) 
            return matrix / np.sum(matrix)

        try:
            stats = get_reference_stats(self.ref_sys_id)
            return stats['Tab_2D_load']
        except ValueError as e:
            print(f"Warning: {e} Defaulting to Reference System 1.")
            stats = get_reference_stats(1)
            return stats['Tab_2D_load']

    def _calculate_total_load(self, loading_level: str, total_gen_capacity: float) -> float:
        """
        Calculates Total Load based on the loading level strategy.
        Strategies:
            'D': Deterministic formula based on N.
            'L': Light loading (30-40% of Gen Capacity).
            'M': Medium loading (50-60% of Gen Capacity).
            'H': Heavy loading (70-80% of Gen Capacity).
        """
        if loading_level == 'D':
            # Formula: 10^(-0.2*(log10(N))^2 + 1.98*log10(N) + 0.58)
            log_n = np.log10(self.n_nodes)
            exponent = -0.2 * (log_n**2) + 1.98 * log_n + 0.58
            return 10**exponent
            
        elif loading_level == 'L':
            factor = 0.3 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        elif loading_level == 'M':
            factor = 0.5 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        elif loading_level == 'H':
            factor = 0.7 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        else:
            raise ValueError("Invalid loading level. Choose 'D', 'L', 'M', or 'H'.")

    def _initial_load_distribution(self, total_load: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Generates load sizes based on empirical distributions.
        99% exponential, 1% super large.
        """
        if self.n_load == 0:
            return np.array([]), 0.0, np.array([])

        mu = total_load / self.n_load
        
        # Base distribution: Exponential
        p_loads = np.random.exponential(scale=mu, size=self.n_load)
        
        # Super large loads (approx 1%)
        n_super = int(round(0.01 * self.n_load))
        if n_super > 0:
            max_p = np.max(p_loads)
            # Uniform random [1*max, 3*max]
            p_super = max_p + (2.0 * np.random.rand(n_super) * max_p)
            
            # Replace random indices
            indices = np.random.choice(self.n_load, size=n_super, replace=False)
            p_loads[indices] = p_super

        # Scaling check
        current_sum = np.sum(p_loads)
        if current_sum > 1.05 * total_load or current_sum < 0.90 * total_load:
            p_loads = p_loads * (total_load / current_sum)
            
        max_r_pl = np.max(p_loads)
        normalized_r_pl = p_loads / max_r_pl
        
        return p_loads, max_r_pl, normalized_r_pl

    def _assignment_logic(self, norm_deg_pairs: np.ndarray, norm_vals: np.ndarray, tab_2d: np.ndarray) -> np.ndarray:
        """
        Generic binning assignment logic (Shared logic with CapacityAllocator).
        """
        n_items = len(norm_vals)
        
        # 1. Calculate target counts
        tab_2d_n = np.round(tab_2d * n_items).astype(int)
        
        diff = n_items - np.sum(tab_2d_n)
        if diff != 0:
            flat_idx = np.argmax(tab_2d_n)
            r_idx, c_idx = np.unravel_index(flat_idx, tab_2d_n.shape)
            tab_2d_n[r_idx, c_idx] += diff

        # 2. Binning Targets
        n_k_targets = np.sum(tab_2d_n, axis=0) # Degree bins (Cols)
        n_v_targets = np.sum(tab_2d_n, axis=1) # Value bins (Rows)
        
        # 3. Sort and Bin Nodes by Degree
        sorted_nodes = norm_deg_pairs[norm_deg_pairs[:, 1].argsort()]
        
        k_bins = []
        current_idx = 0
        for count in n_k_targets:
            end_idx = current_idx + count
            k_bins.append(sorted_nodes[current_idx:end_idx].copy())
            current_idx = end_idx
            
        # 4. Sort and Bin Values (Loads)
        sorted_vals = np.sort(norm_vals)
        
        v_bins = []
        current_idx = 0
        for count in n_v_targets:
            end_idx = current_idx + count
            v_bins.append(list(sorted_vals[current_idx:end_idx]))
            current_idx = end_idx

        # 5. Assignment Loop
        for k_idx in range(14): 
            for v_idx in range(13, -1, -1):
                count = tab_2d_n[v_idx, k_idx]
                if count > 0:
                    vals_to_assign = []
                    source_bin = v_bins[v_idx]
                    take = min(count, len(source_bin))
                    
                    for _ in range(take):
                        rand_i = np.random.randint(0, len(source_bin))
                        vals_to_assign.append(source_bin.pop(rand_i))
                    
                    if k_bins[k_idx].shape[1] < 3:
                        k_bins[k_idx] = np.hstack((k_bins[k_idx], np.full((len(k_bins[k_idx]), 1), -1.0)))
                        
                    empty_slots = np.where(k_bins[k_idx][:, 2] == -1.0)[0]
                    fill_indices = empty_slots[:take]
                    k_bins[k_idx][fill_indices, 2] = vals_to_assign

        final_list = []
        for bin_arr in k_bins:
            if bin_arr.shape[1] == 3:
                final_list.append(bin_arr)
                
        if not final_list:
            return np.array([])
            
        return np.vstack(final_list)

    def allocate(self, loading_level: str = 'H') -> Dict[int, float]:
        """
        Allocates loads to buses.
        
        Args:
            loading_level: 'D' (Default Formula), 'L' (Light), 'M' (Medium), 'H' (Heavy).
            
        Returns:
            Dictionary mapping Load Node ID -> Active Power (MW)
        """
        if self.n_load == 0:
            print("No load buses found. Skipping load allocation.")
            return {}

        # 1. Get Total Generation Capacity from Grid
        gen_caps = [d.get('pg_max', 0.0) for n, d in self.graph.nodes(data=True) if d.get('bus_type') == 'Gen']
        total_gen_capacity = sum(gen_caps)
        
        if total_gen_capacity == 0 and loading_level != 'D':
            print("Warning: Total generation capacity is 0. Switching to 'D' (Deterministic) loading.")
            loading_level = 'D'

        # 2. Calculate Total Load
        total_load = self._calculate_total_load(loading_level, total_gen_capacity)
        
        print(f"Allocating Loads for {self.n_load} load buses.")
        print(f"Total System Load Target: {total_load:.2f} MW (Level: {loading_level})")

        # 3. Get Load Bus Degrees
        load_degrees = []
        for n in self.load_buses:
            load_degrees.append([n, self.graph.degree(n)])
        load_degrees = np.array(load_degrees)
        
        if len(load_degrees) == 0:
            return {}

        max_deg = np.max(load_degrees[:, 1])
        norm_load_degrees = load_degrees.copy().astype(float)
        norm_load_degrees[:, 1] = norm_load_degrees[:, 1] / max_deg

        # 4. Initial Load Distribution
        p_loads_actual, max_r_pl, norm_r_pl = self._initial_load_distribution(total_load)
        
        # 5. Get Reference Table
        tab_2d_load = self._get_tab_2d_load()
        
        # 6. Assignment
        norm_assignment = self._assignment_logic(norm_load_degrees, norm_r_pl, tab_2d_load)
        
        # 7. Denormalize
        final_allocation = {}
        for row in norm_assignment:
            node_id = int(row[0])
            norm_val = row[2]
            actual_val = norm_val * max_r_pl
            final_allocation[node_id] = actual_val
            
        return final_allocation