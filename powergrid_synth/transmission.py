import numpy as np
import networkx as nx
from scipy.stats import levy_stable
from typing import Dict, Tuple, List
from .reference_data import get_reference_stats
from .dcpf import DCPowerFlow

class TransmissionLineAllocator:
    """
    Allocates impedance (X, R) and Capacity Limits to transmission lines.
    Based on 'sg_flow_lim.m' from SynGrid.
    
    Steps:
    1. Initialize random impedances (Zpr) based on LogNormal distribution.
    2. Run iterative DCPF (Swapping Logic):
       - Calculate flows.
       - Assign lower Impedance (Z) to lines with higher Flow.
       - Perform random swaps to introduce variance.
    3. (Optional) Topology Refinement:
       - Add low-impedance lines to bridge large phase angle differences.
       - Remove weak (high-impedance) lines to maintain grid density.
    4. Allocate Capacity:
       - Use 'Tab_2D_FlBeta' to assign Capacity Factors (Beta).
       - Capacity = Flow / Beta.
    """

    def __init__(self, graph: nx.Graph, ref_sys_id: int = 1):
        self.graph = graph
        self.ref_sys_id = ref_sys_id
        
        try:
            self.stats = get_reference_stats(ref_sys_id)
        except ValueError:
             self.stats = get_reference_stats(1)
             
        self.stab_params = self.stats.get('stab', [1.374, -0.838, 2.965, 85.801])
        self.tab_fl_beta = self.stats['Tab_2D_FlBeta']
        self.mu_beta = self.stats.get('mu_beta', 0.27)
        self.overload_b = self.stats.get('Overload_b', 0.00083)

    def _generate_phi(self, num_lines: int) -> np.ndarray:
        """Generates line angles (phi) using stable distribution."""
        alpha, beta, gamma, delta = self.stab_params
        
        # sg_random_stable implementation via scipy
        phi = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=num_lines)
        
        # Ensure phi is within logical bounds [0, 90] degrees mostly
        # MATLAB loops indefinitely until valid; we use a cap to prevent infinite loops
        mask = (phi > 90) | (phi < 0)
        max_retries = 20
        count = 0
        while np.any(mask) and count < max_retries:
            n_resample = np.sum(mask)
            new_vals = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_resample)
            phi[mask] = new_vals
            mask = (phi > 90) | (phi < 0)
            count += 1
            
        # Hard clip fallback if retries exhausted
        phi = np.clip(phi, 0.01, 89.99)
        return phi

    def _generate_beta(self, num_lines: int) -> np.ndarray:
        """Generates capacity factors (beta) using exponential distribution."""
        # sg_exprnd -> numpy.random.exponential (scale = mean)
        beta = np.random.exponential(scale=self.mu_beta, size=num_lines)
        
        # Handle outliers (beta > 1)
        mask = beta > 1.0
        max_retries = 10
        count = 0
        while np.any(mask) and count < max_retries:
            n_resample = np.sum(mask)
            beta[mask] = np.random.exponential(scale=self.mu_beta, size=n_resample)
            mask = beta > 1.0
            count += 1
            
        # Overload injection (simulating bottleneck lines)
        n_overload = int(round(num_lines * self.overload_b))
        if n_overload > 0:
            indices = np.random.choice(num_lines, n_overload, replace=False)
            # Overload_bet = 1 + 0.2 * rand
            beta[indices] = 1.0 + 0.2 * np.random.rand(n_overload)
            
        # Avoid zero or near-zero betas (division by zero protection)
        small_mask = beta < 1e-4
        n_small = np.sum(small_mask)
        if n_small > 0:
            beta[small_mask] = 0.01 + 0.099 * np.random.rand(n_small)
        
        return np.sort(beta)

    def _assign_betas(self, flows: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """
        Assigns Beta factors to lines based on Flow magnitude using 2D Table.
        Implements 'Assignment_Fl' logic.
        """
        num_lines = len(flows)
        if num_lines == 0: return np.array([])

        tab_n = np.round(self.tab_fl_beta * num_lines).astype(int)
        
        # Adjust for rounding errors to match total lines
        diff = num_lines - np.sum(tab_n)
        if diff != 0:
            flat_idx = np.argmax(tab_n)
            r, c = np.unravel_index(flat_idx, tab_n.shape)
            tab_n[r, c] += diff

        # Targets per Flow class (Rows) and Beta class (Cols)
        n_flow_targets = np.sum(tab_n, axis=1) 
        n_beta_targets = np.sum(tab_n, axis=0) 
        
        # Sort flows (input is [idx, val])
        sorted_flows = flows[flows[:, 1].argsort()]
        
        # Binning Flows
        flow_bins = []
        curr = 0
        for count in n_flow_targets:
            end = curr + count
            flow_bins.append(list(sorted_flows[curr:end]))
            curr = end
            
        # Binning Betas (already sorted)
        beta_bins = [] 
        curr = 0
        for count in n_beta_targets:
            end = curr + count
            beta_bins.append(list(betas[curr:end]))
            curr = end
            
        # Assignment Loop: Rows(Flows) x Cols(Betas)
        final_assignment = [] # [LineIdx, FlowVal, BetaVal]
        
        # Iterate Rows (Flows) - typically high to low in MATLAB (14:-1:1)
        for r in range(len(n_flow_targets)-1, -1, -1):
            # Iterate Cols (Betas) - typically high to low in MATLAB (16:-1:1)
            for c in range(len(n_beta_targets)-1, -1, -1):
                count = tab_n[r, c]
                if count > 0:
                    f_bin = flow_bins[r]
                    b_bin = beta_bins[c]
                    
                    take = min(count, len(f_bin), len(b_bin))
                    
                    for _ in range(take):
                        if f_bin and b_bin:
                            # Random sample without replacement
                            f_idx = np.random.randint(0, len(f_bin))
                            flow_data = f_bin.pop(f_idx) # [LineIdx, FlowVal]
                            
                            b_idx = np.random.randint(0, len(b_bin))
                            beta_val = b_bin.pop(b_idx)
                            
                            final_assignment.append([flow_data[0], flow_data[1], beta_val])

        # Handle leftovers if any (due to bin mismatch/emptying)
        rem_flows = [f for bin in flow_bins for f in bin]
        rem_betas = [b for bin in beta_bins for b in bin]
        
        for i in range(min(len(rem_flows), len(rem_betas))):
             final_assignment.append([rem_flows[i][0], rem_flows[i][1], rem_betas[i]])
             
        return np.array(final_assignment)

    def _refine_topology(self):
        """
        Refines grid topology to reduce max phase angle difference.
        Matches 'sg_flow_lim.m' heuristic:
          1. Check max angle difference.
          2. Add line between max diff pair.
          3. Remove a weak line to maintain density.
        """
        n_nodes = self.graph.number_of_nodes()
        # Target threshold for angle difference
        tt_target = 10**(0.3196 * np.log10(n_nodes) + 0.8324)
        
        max_iter = 10 # Safety limit to prevent infinite loops
        
        for _ in range(max_iter):
            dcpf = DCPowerFlow(self.graph)
            flows, angles = dcpf.run() 
            
            theta_rad = np.array(list(angles.values()))
            node_keys = list(angles.keys())
            
            # Convert to degrees
            theta_deg = np.degrees(theta_rad)
            if len(theta_deg) == 0: break 
            
            angle_spread = np.max(theta_deg) - np.min(theta_deg)
            
            # Condition: max(abs(DD)) > TT_target + 2
            if angle_spread <= tt_target + 2:
                break # Grid is sufficiently compact electrically
                
            # --- 1. Add Strengthening Line ---
            # Find pair with max difference
            min_idx = np.argmin(theta_deg)
            max_idx = np.argmax(theta_deg)
            min_node = node_keys[min_idx]
            max_node = node_keys[max_idx]
            
            # Add edge if not exists
            if not self.graph.has_edge(min_node, max_node):
                # Low impedance parameters from MATLAB code
                r_new = 0.001 + np.random.rand() * 0.001
                x_new = 0.002 + np.random.rand() * 0.003
                z_new = np.sqrt(r_new**2 + x_new**2)
                
                self.graph.add_edge(min_node, max_node, r=r_new, x=x_new, z=z_new)
            
            # --- 2. Remove Weak Line ---
            # Candidates: Select from lines with high X (Impedance)
            edges = list(self.graph.edges(data=True))
            if not edges: break

            # Sort by X descending (High X = Weak/Long line)
            sorted_edges = sorted(edges, key=lambda e: e[2].get('x', 0), reverse=True)
            
            n_edges = len(sorted_edges)
            br_sl = 0.8 # Select top 20%
            n_candidates = int(n_edges * (1.0 - br_sl))
            n_candidates = max(n_candidates, 1)
            
            # Candidates are the 'tail' of the list in MATLAB (high X)
            candidates = sorted_edges[:n_candidates]
            
            # Filter by degree to avoid isolating nodes
            valid_candidates = []
            min_deg = 3 if n_nodes >= 40 else 2
            
            for u, v, d in candidates:
                if self.graph.degree(u) >= min_deg and self.graph.degree(v) >= min_deg:
                    valid_candidates.append((u, v, d))
            
            if valid_candidates:
                # Pick random candidate to remove
                idx = np.random.randint(0, len(valid_candidates))
                u_rem, v_rem, d_rem = valid_candidates[idx]
                
                self.graph.remove_edge(u_rem, v_rem)
                
                # Verify Grid Integrity
                # MATLAB: checks success flag. Here we check connectivity.
                if not nx.is_connected(self.graph):
                    # Revert if disconnected
                    self.graph.add_edge(u_rem, v_rem, **d_rem)

    def allocate(self, refine_topology: bool = False) -> Dict[Tuple[int, int], float]:
        """
        Main execution method.
        """
        edges = list(self.graph.edges())
        m_lines = len(edges)
        n_nodes = self.graph.number_of_nodes()
        
        if m_lines == 0: return {}

        # 1. Initial Impedance (Zpr)
        # LogNormal distribution
        zpr_pars = self.stats.get('Zpr_pars', [-2.38, 1.99, 1.99])
        mu_len = zpr_pars[0]
        sigma_len = zpr_pars[1]
        
        zpr = np.random.lognormal(mu_len, sigma_len, m_lines)
        # physical practice in the per-unit system, and for stability 
        zpr = np.clip(zpr, 0.001, 0.5) 
        
        # 2. Generate Phi and Initial X, R
        phi = self._generate_phi(m_lines)
        
        for i, (u, v) in enumerate(edges):
            # X = Z * sin(phi), R = Z * cos(phi)
            # Zpr is magnitude
            x_val = zpr[i] * np.sin(np.deg2rad(phi[i]))
            r_val = zpr[i] * np.cos(np.deg2rad(phi[i]))
            self.graph[u][v]['x'] = max(x_val, 1e-5)
            self.graph[u][v]['r'] = max(r_val, 1e-5)
            self.graph[u][v]['z'] = zpr[i] 
            self.graph[u][v]['edge_idx'] = i

        # 3. Iterative Refinement (The 'kk' loop)
        iterations = 2 if n_nodes >= 300 else 1
        dcpf = DCPowerFlow(self.graph)
        
        for _ in range(iterations):
            # Calculate Flow
            flows, _ = dcpf.run()
            
            # Map flows to current edges
            flow_vals = np.zeros(m_lines)
            for i, (u, v) in enumerate(edges):
                flow_vals[i] = abs(flows.get((u, v), 0.0))
            
            # Sort Z magnitudes (Ascending)
            current_z = np.array([self.graph[u][v]['z'] for u, v in edges])
            sorted_z = np.sort(current_z) 
            
            # Sort Flows (Descending)
            # We want High Flow to get Low Z
            flow_indices = np.argsort(-flow_vals) 
            
            # Create a pool of Z values ordered by flow rank
            # z_pool[0] is smallest Z, assigned to flow_indices[0] (highest flow)
            z_pool = sorted_z.copy()
            
            # Swapping Logic
            if n_nodes > 1200:
                as_param = 0.3
                an_param = 0.8
            else:
                as_param = 0.2
                an_param = 0.2
                
            n_swap = int(round(as_param * m_lines))
            n_neib = int(round(an_param * m_lines))
            
            for _ in range(n_swap):
                # Pick random index
                idx1 = np.random.randint(0, m_lines)
                if idx1 == 0: idx1 = 1 
                
                # Pick neighbor
                xch_disf = int(np.floor(n_neib * np.random.rand()))
                xch_dis = min(xch_disf, (m_lines - 1 - idx1))
                idx2 = idx1 + xch_dis
                
                # Swap Z values
                if idx2 < m_lines:
                    z_pool[idx1], z_pool[idx2] = z_pool[idx2], z_pool[idx1]
            
            # Assign Z back to lines
            for rank_i, line_idx in enumerate(flow_indices):
                z_val = z_pool[rank_i]
                u, v = edges[line_idx]
                
                # Recalculate X, R using original Phi structure
                p = phi[line_idx]
                self.graph[u][v]['z'] = z_val
                self.graph[u][v]['x'] = max(z_val * np.sin(np.deg2rad(p)), 1e-5)
                self.graph[u][v]['r'] = max(z_val * np.cos(np.deg2rad(p)), 1e-5)

        # 4. Topology Refinement (Optional)
        if refine_topology:
            self._refine_topology()
            # Refresh edge list after topology changes
            edges = list(self.graph.edges())
            m_lines = len(edges)

        # 5. Final Flow Calculation
        flows, _ = dcpf.run()
        flow_vals = np.array([abs(flows.get((u, v), 0.0)) for u, v in edges])
        max_flow = np.max(flow_vals) if len(flow_vals) > 0 and np.max(flow_vals) > 0 else 1.0
        
        # 6. Generate & Assign Capacity Factors (Beta)
        betas = self._generate_beta(m_lines)
        
        # Normalize flows
        norm_flow_data = []
        for i, val in enumerate(flow_vals):
            norm_flow_data.append([i, val / max_flow])
        norm_flow_data = np.array(norm_flow_data)
        
        # Assign Betas
        assigned = self._assign_betas(norm_flow_data, betas)
        
        # 7. Calculate & Set Limits
        line_caps = {}
        for row in assigned:
            idx = int(row[0])
            beta = row[2]
            u, v = edges[idx]
            
            flow_mw = flow_vals[idx]
            if beta < 1e-3: beta = 1e-3
            limit = flow_mw / beta
            
            # Min capacity check (MATLAB: <=2 -> 5 + 100*rand)
            if limit <= 2.0: 
                limit = 5.0 + 100 * np.random.rand() 
            
            self.graph[u][v]['capacity'] = limit
            line_caps[(u, v)] = limit
            
        return line_caps