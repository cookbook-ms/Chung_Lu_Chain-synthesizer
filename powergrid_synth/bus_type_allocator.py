r"""
This module provides the :class:`BusTypeAllocator` algorithm, a class for assigning bus types in a grid topology. Read more in :doc:`Bus Type Assignment based on Bus Type Entropy </theory/bus_type_assignment>`.
"""

import numpy as np
import networkx as nx
import math
import random
from typing import List, Dict, Tuple, Optional


class BusTypeAllocator:
    r"""
    This class assigns bus types (Generator, Load, Connection) to a raw power grid topology
    using an Artificial Immune System (AIS) optimization algorithm to match
    target topological entropy properties.

    Args:
        graph: NetworkX graph representing the grid topology.
        entropy_model: 0 or 1, determines the entropy definition used (W parameter).
    """
    
    TYPE_GEN = 1
    TYPE_LOAD = 2
    TYPE_CONN = 3

    def __init__(self, graph: nx.Graph, entropy_model: int = 0):
        self.graph = graph
        self.entropy_model = entropy_model
        
        # Pre-calculate graph stats
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.n_edges = graph.number_of_edges()
        
        # Map node to 0-based index for internal matrix logic
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes)}
        
        # Edge list as indices (M x 2)
        self.link_ids = []
        for u, v in graph.edges():
            self.link_ids.append([self.node_to_idx[u], self.node_to_idx[v]])
        self.link_ids = np.array(self.link_ids)

        # Pre-calculate degrees for constraint checking (Connection buses shouldn't be leaves)
        self.degrees = np.array([graph.degree(n) for n in self.nodes])
        self.non_leaf_indices = np.where(self.degrees > 1)[0]

        # Determine Ratios [G, L, C] based on network size
        self.ratio_types = self._get_ratios(self.n_nodes)
        
        # Store entropy samples for plotting
        self.w_samples: List[float] = []
        
    def _get_ratios(self, n: int) -> List[float]:
        r"""
        Bus type G/L/C ratio settings based on network size.
        """
        if n < 2000:
            return [0.23, 0.55, 0.22] # IEEE-300 like
        elif n < 10000:
            return [0.33, 0.44, 0.23] # NYISO like
        else:
            return [0.2, 0.4, 0.4]    # WECC like

    def _get_d_parameter(self, n: int) -> float:
        """Calculates the normalized parameter for W_star estimation."""
        log_n = math.log(n)
        if self.entropy_model == 0:
            if log_n <= 8:
                return -1.39 * log_n + 6.79
            else:
                return -6.003e-14 * (log_n**15.48)
        else:
            if log_n <= 8:
                return -1.748 * log_n + 8.576
            else:
                return -6.053e-22 * (log_n**24.1)

    def _generate_random_assignment(self) -> np.ndarray:
        """
        Generates a valid random bus type assignment vector (1xN).
        1=Gen, 2=Load, 3=Conn.
        Constraint: Type 3 (Conn) should prefer non-leaf nodes (degree > 1).
        """
        assignment = np.zeros(self.n_nodes, dtype=int)
        
        n_gen = int(round(self.n_nodes * self.ratio_types[0]))
        n_conn = int(round(self.n_nodes * self.ratio_types[2]))
        # Load is remainder
        
        # 1. Assign Connection Buses (Type 3)
        # Prefer nodes with degree > 1
        candidates_conn = list(self.non_leaf_indices)
        if len(candidates_conn) < n_conn:
            # Fallback if not enough non-leafs
            candidates_conn = list(range(self.n_nodes))
            
        chosen_conn = np.random.choice(candidates_conn, size=n_conn, replace=False)
        assignment[chosen_conn] = self.TYPE_CONN
        
        # 2. Assign Generator Buses (Type 1)
        # Choose from remaining empty spots (0)
        available_indices = np.where(assignment == 0)[0]
        if len(available_indices) > 0:
            count = min(n_gen, len(available_indices))
            chosen_gen = np.random.choice(available_indices, size=count, replace=False)
            assignment[chosen_gen] = self.TYPE_GEN
            
        # 3. Assign Load Buses (Type 2) to remainder
        assignment[assignment == 0] = self.TYPE_LOAD
        
        return assignment

    def _calculate_link_ratios(self, assignment: np.ndarray) -> List[float]:
        r"""
        Calculates ratios of the 6 link types: GG, LL, CC, GL, GC, LC.
        Encode links using the follwoing mappings:
        
        .. math::
            11 \to \text{GG}, 22 \to \text{LL}, 33 \to \text{CC}, 
            12 \to \text{GL}, 13 \to \text{GC}, 23 \to \text{LC}
        """
        # Vectorized link type check
        u_types = assignment[self.link_ids[:, 0]]
        v_types = assignment[self.link_ids[:, 1]]
        
        # Sort pair so (1,2) is same as (2,1)
        pairs = np.sort(np.vstack((u_types, v_types)).T, axis=1)

        counts = {
            (1, 1): 0, (2, 2): 0, (3, 3): 0,
            (1, 2): 0, (1, 3): 0, (2, 3): 0
        }
        
        # Count unique pairs using numpy
        unique, counts_arr = np.unique(pairs, axis=0, return_counts=True)
        for pair, count in zip(unique, counts_arr):
            counts[tuple(pair)] = count
            
        M = self.n_edges if self.n_edges > 0 else 1
        return [
            counts[(1, 1)] / M, # GG
            counts[(2, 2)] / M, # LL
            counts[(3, 3)] / M, # CC
            counts[(1, 2)] / M, # GL
            counts[(1, 3)] / M, # GC
            counts[(2, 3)] / M  # LC
        ]

    def _calculate_entropy_score(self, bus_ratios: List[float], link_ratios: List[float]) -> float:
        """Calculates W score based on entropy model."""
        EPS = 1e-12 # Avoid log(0)
        
        if self.entropy_model == 0:
            # Model 0: - sum(p log p)
            x1 = sum(r * math.log(r + EPS) for r in bus_ratios)
            x2 = sum(r * math.log(r + EPS) for r in link_ratios if r > 0)
        else:
            # Model 1: Weighted by N and M
            x1 = sum(math.log(r + EPS) * (r * self.n_nodes) for r in bus_ratios)
            x2 = sum(math.log(r + EPS) * (r * self.n_edges) for r in link_ratios if r > 0)
            
        return -(x1 + x2)

    def _estimate_w_star(self, monte_carlo_iters: int = 10_000) -> Tuple[float, float]:
        r"""
        Runs Monte Carlo simulation to estimate target W* value.
        Returns:
            Tuple of (W_star, Standard_Deviation_of_W)
        """
        w_samples = []
        for _ in range(monte_carlo_iters):
            assign = self._generate_random_assignment()
            l_ratios = self._calculate_link_ratios(assign)
            w = self._calculate_entropy_score(self.ratio_types, l_ratios)
            w_samples.append(w)
        
        # Store for plotting
        self.w_samples = w_samples
            
        mean_w = np.mean(w_samples)
        std_w = np.std(w_samples)
        d_param = self._get_d_parameter(self.n_nodes)
        
        w_star = (d_param * std_w) + mean_w
        return w_star, std_w

    def allocate(self, max_iter: int = 100, population_size: int = 20) -> Dict[int, str]:
        r"""
        Main AIS optimization method.
        
        Improved based on SynGrid 'sg_bus_type.m':
        1. Uses dynamic convergence criteria based on MC std deviation.
        2. Implements Clonal and Mutation operators with rank-based intensity.
        3. Injects fresh random solutions every iteration for diversity.

        Args:
            max_iter: Maximum optimization iterations.
            population_size: Size of the surviving population (default 20, as in MATLAB).
        """
        print(f"Starting Bus Type Allocation (N={self.n_nodes}, M={self.n_edges})...")
        
        # 1. Estimate Target W* and Standard Deviation
        w_star, std_w = self._estimate_w_star()
        print(f"  Target Entropy Score (W*): {w_star:.4f}, Std Dev: {std_w:.4f}")
        
        # Determine Convergence Criteria (Logic from MATLAB)
        if self.n_nodes < 50:
            criteria = std_w / 2 if self.entropy_model == 1 else std_w / 10
        else:
            criteria = std_w / 1000

        # 2. Initialize Population
        population = []
        for _ in range(population_size):
            population.append(self._generate_random_assignment())
            
        best_solution = None
        best_error = float('inf')
        
        # 3. Optimization Loop
        for it in range(max_iter):
            # A. Evaluate Fitness
            scores = []
            for indiv in population:
                l_ratios = self._calculate_link_ratios(indiv)
                w = self._calculate_entropy_score(self.ratio_types, l_ratios)
                error = abs(w_star - w)
                scores.append((error, indiv))
            
            # Sort by error (Ascending)
            scores.sort(key=lambda x: x[0])
            
            # Update Global Best
            current_best_error, current_best_sol = scores[0]
            if current_best_error < best_error:
                best_error = current_best_error
                best_solution = current_best_sol
            
            # Check Convergence
            if best_error < criteria:
                print(f"  Converged at iteration {it}. Error: {best_error:.6f} < Criteria: {criteria:.6f}")
                break
            
            if it % 10 == 0:
                print(f"  Iter {it}: Best Error = {best_error:.6f}")

            # B. Clonal Selection
            # Select top solutions to clone. 
            # Logic: B=0.4, s=100. nc = round(B*s/rank).
            # MATLAB uses the whole population here as source? Yes, iterates 1:L.
            clones = []
            B, s = 0.4, 100
            
            # Limit cloning to existing population size (sorted)
            for rank, item in enumerate(scores):
                indiv = item[1]
                # Rank is 1-based in formula
                n_clones = int(round((B * s) / (rank + 1)))
                for _ in range(n_clones):
                    clones.append(indiv.copy())
            
            # C. Mutation Operator
            # Mutates the clones.
            # Logic: Split clones into 10 groups. 
            # Worse groups get MORE mutations (higher intensity).
            mutated_clones = []
            L_clones = len(clones)
            chunk_size = L_clones / 10.0
            
            for j in range(L_clones):
                clone = clones[j]
                
                # Determine "group" (b) from 1 to 10
                # j starts at 0 (best clones) to L_clones-1 (worst clones)
                # Actually, clones are generated from best to worst parents, so order is roughly preserved.
                group_idx = int(j / chunk_size) + 1 # 1 to 10
                if group_idx > 10: group_idx = 10
                
                # Apply mutation 'group_idx' times
                for _ in range(group_idx):
                    # Pick random node
                    node_idx = np.random.randint(0, self.n_nodes)
                    
                    # Random flip to any other type (1, 2, or 3)
                    # MATLAB code bias: if rand<0.5 -> 1 else -> 2. 
                    # We will allow all 3 to maintain topological diversity.
                    new_type = np.random.choice([1, 2, 3])
                    clone[node_idx] = new_type
                
                mutated_clones.append(clone)
            
            # D. Diversity Injection
            # Add fresh random solutions. MATLAB adds 10% of new population size.
            n_fresh = max(1, int(len(mutated_clones) / 10))
            fresh_solutions = []
            for _ in range(n_fresh):
                fresh_solutions.append(self._generate_random_assignment())
                
            # E. Selection (Next Generation)
            # Combine: Fresh + Mutated Clones + Original Clones (MATLAB logic combines pop4, pop3, pop2)
            combined_population = fresh_solutions + mutated_clones + clones
            
            # Re-evaluate Combined Population
            combined_scores = []
            for indiv in combined_population:
                l_ratios = self._calculate_link_ratios(indiv)
                w = self._calculate_entropy_score(self.ratio_types, l_ratios)
                err = abs(w_star - w)
                combined_scores.append((err, indiv))
            
            # Sort and Truncate to original population size
            combined_scores.sort(key=lambda x: x[0])
            
            # Keep top 'population_size'
            population = [x[1] for x in combined_scores[:population_size]]

        # 4. Final mapping
        bus_types = {}
        type_labels = {1: 'Gen', 2: 'Load', 3: 'Conn'}
        # Use best_solution found across all iterations
        for i, type_code in enumerate(best_solution):
            node_id = self.idx_to_node[i]
            bus_types[int(node_id)] = type_labels[type_code]
            
        return bus_types

    def plot_entropy_pdf(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plots the empirical Probability Density Function (PDF) of the entropy values (W) 
        gathered during estimation, optionally fitting a Normal Distribution.
        
        Must be called after `allocate()` or `_estimate_w_star()`.
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import norm
        except ImportError:
            print("Matplotlib or Scipy not installed. Cannot plot PDF.")
            return

        if not hasattr(self, 'w_samples') or not self.w_samples:
            print("No entropy samples available. Please run allocate() first.")
            return

        plt.figure(figsize=figsize)
        
        # Plot Histogram
        # density=True ensures the area under the histogram sums to 1 (probability density)
        count, bins, ignored = plt.hist(self.w_samples, bins=50, density=True, 
                                      alpha=0.6, color='skyblue', edgecolor='black', 
                                      label='Empirical Data')
        
        # Fit and plot Normal Distribution
        mu, std = norm.fit(self.w_samples)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        
        plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit\n$\mu={mu:.3f}$, $\sigma={std:.3f}$')
        
        plt.title(f'Bus Type Entropy Distribution (N={self.n_nodes})')
        plt.xlabel('Bus Type Entropy (W)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()