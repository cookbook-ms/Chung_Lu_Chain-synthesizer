import numpy as np
import networkx as nx
import math
import random
from typing import List, Dict, Tuple, Optional

class BusTypeAllocator:
    """
    Assigns bus types (Generator, Load, Connection) to a power grid topology
    using an Artificial Immune System (AIS) optimization algorithm to match
    target topological entropy properties.
    
    Ported and adapted from 'sg_bus_type.m' (SynGrid).
    """
    
    TYPE_GEN = 1
    TYPE_LOAD = 2
    TYPE_CONN = 3

    def __init__(self, graph: nx.Graph, entropy_model: int = 0):
        """
        Args:
            graph: NetworkX graph representing the grid topology.
            entropy_model: 0 or 1, determines the entropy definition used (W parameter).
        """
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
        
    def _get_ratios(self, n: int) -> List[float]:
        """Defined in code: G/L/C ratios based on network size."""
        if n < 2000:
            return [0.23, 0.55, 0.22] # IEEE-300 like
        elif n < 10000:
            return [0.33, 0.44, 0.23] # NYISO like
        else:
            return [0.2, 0.4, 0.4]    # WECC like

    def _get_d_parameter(self, n: int) -> float:
        """Calculates distance parameter for W_star estimation."""
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
        
        # Counts based on ratios
        # Note: The MATLAB code had a specific logic where loop 1 used Ratio(2) for Type 3.
        # Here we implement the logical mapping: Ratio[0]->Gen, Ratio[1]->Load, Ratio[2]->Conn.
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
        """Calculates ratios of the 6 link types: GG, LL, CC, GL, GC, LC."""
        # Vectorized link type check
        u_types = assignment[self.link_ids[:, 0]]
        v_types = assignment[self.link_ids[:, 1]]
        
        # Sort pair so (1,2) is same as (2,1)
        pairs = np.sort(np.vstack((u_types, v_types)).T, axis=1)
        
        # Encode pairs: 11(GG), 22(LL), 33(CC), 12(GL), 13(GC), 23(LC)
        # Mappings:
        # 1-1 -> GG
        # 2-2 -> LL
        # 3-3 -> CC
        # 1-2 -> GL
        # 1-3 -> GC
        # 2-3 -> LC
        
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

    def _estimate_w_star(self, monte_carlo_iters: int = 2000) -> float:
        """
        Runs Monte Carlo simulation to estimate target W* value.
        """
        w_samples = []
        for _ in range(monte_carlo_iters):
            assign = self._generate_random_assignment()
            l_ratios = self._calculate_link_ratios(assign)
            w = self._calculate_entropy_score(self.ratio_types, l_ratios)
            w_samples.append(w)
            
        mean_w = np.mean(w_samples)
        std_w = np.std(w_samples)
        d_param = self._get_d_parameter(self.n_nodes)
        
        return (d_param * std_w) + mean_w

    def allocate(self, max_iter: int = 100, population_size: int = 20) -> Dict[int, str]:
        """
        Main execution method. Runs the AIS optimization.
        
        Returns:
            Dictionary mapping node_id -> 'Gen', 'Load', or 'Conn'
        """
        print(f"Starting Bus Type Allocation (N={self.n_nodes}, M={self.n_edges})...")
        
        # 1. Target W*
        w_star = self._estimate_w_star()
        print(f"  Target Entropy Score (W*): {w_star:.4f}")
        
        # 2. Initialize Population
        population = []
        for _ in range(population_size):
            population.append(self._generate_random_assignment())
            
        best_solution = None
        best_error = float('inf')
        
        # 3. Optimization Loop
        for it in range(max_iter):
            # Evaluate Fitness (Error = abs(W* - W))
            scores = []
            for indiv in population:
                l_ratios = self._calculate_link_ratios(indiv)
                w = self._calculate_entropy_score(self.ratio_types, l_ratios)
                error = abs(w_star - w)
                scores.append((error, indiv))
            
            # Sort by error (Ascending)
            scores.sort(key=lambda x: x[0])
            
            current_best_error, current_best_sol = scores[0]
            if current_best_error < best_error:
                best_error = current_best_error
                best_solution = current_best_sol
            
            # Convergence check (Thresholds from MATLAB code)
            threshold = 0.001 # Simplified threshold
            if best_error < threshold:
                print(f"  Converged at iteration {it}. Error: {best_error:.6f}")
                break
                
            # --- Clonal Selection & Mutation ---
            # Select top half to clone
            top_half = [s[1] for s in scores[:population_size//2]]
            new_population = []
            
            # Cloning & Mutating
            # Better solutions get cloned more, but mutated less? 
            # MATLAB logic: 
            #   Cloning: Count decreases with rank.
            #   Mutation: Probability increases with rank (worse solutions mutated more).
            
            B, s = 0.4, 100
            
            for rank, indiv in enumerate(top_half):
                # Number of clones
                n_clones = int(round(B * s / (rank + 1)))
                
                for _ in range(n_clones):
                    clone = indiv.copy()
                    
                    # Mutation
                    # Logic adapted: rank 0 (best) mutates little, rank N mutates more
                    # Rate increases with rank
                    mutation_rate = 0.01 + (rank / len(top_half)) * 0.1
                    
                    if np.random.random() < mutation_rate:
                        # Pick random node and change type
                        idx_to_mut = np.random.randint(0, self.n_nodes)
                        # Pick new type (1, 2, or 3)
                        new_type = np.random.choice([1, 2, 3])
                        clone[idx_to_mut] = new_type
                        
                    new_population.append(clone)
            
            # Fill remainder with random new solutions (Diversity)
            while len(new_population) < population_size:
                new_population.append(self._generate_random_assignment())
                
            # Truncate if too huge (optimization)
            if len(new_population) > population_size * 5:
                new_population = new_population[:population_size*5]
                
            population = new_population
            
            if it % 10 == 0:
                print(f"  Iter {it}: Best Error = {best_error:.6f}")

        # 4. Final mapping
        result_mapping = {}
        type_labels = {1: 'Gen', 2: 'Load', 3: 'Conn'}
        for i, type_code in enumerate(best_solution):
            node_id = self.idx_to_node[i]
            result_mapping[node_id] = type_labels[type_code]
            
        return result_mapping