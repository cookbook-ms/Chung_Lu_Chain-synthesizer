r"""
This calss assists the operation mode II in grid topology generation, where one takes as inputs 

1. `n`: the number of nodes; `ave_k`: the average node degree; and `diam`: the desired diameter for same-voltage level and the node degree distribution types

- `dgln`: discrete generalized log-normal distribution 
- `dpl`: discrete power-law distribution 

2. the transformer line specs between different-voltage levels: `type`: `k-stars`, parameters `c` and `gamma` for the transformer degree distribution.

Example
-------
.. code-block:: python
   :linenos:
   
    # Define the voltage levels (Node count, Avg Degree, Diameter, Distribution Type)
    level_specs = [
        {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},    # (Log-Normal)
        {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},     # (Power Law)
        {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'}  # (Poisson)
    ]
    # Define connections between levels (k-stars model)
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }
    # Initialize Configurator
    configurator = InputConfigurator(seed=100)
    # Generating Input Parameters
    params = configurator.create_params(level_specs, connection_specs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from .deg_dist_optimizer import DegreeDistributionOptimizer

class InputConfigurator:
    """
    Helper class to artificially generate the detailed input sequences (degrees, transformer connections)
    required by PowerGridGenerator from high-level parameters like the number of vertices, and some hyperparameters for the used fitting functions or distributions.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.optimizer = DegreeDistributionOptimizer(verbose=False)

    def _generate_poisson_degrees(self, n_nodes: int, avg_degree: float) -> List[int]:
        """Generates degree sequence using Poisson distribution."""
        return self.rng.poisson(avg_degree, n_nodes).tolist()

    def _generate_optimized_degrees(self, n_nodes: int, avg_degree: float, 
                                    max_degree: int, dist_type: str) -> List[int]:
        """
        Generates degrees by first optimizing parameters for DGLN/DPL to match 
        the target average, then sampling from that distribution.
        
        Args: 
            n_nodes: Number of nodes in each same-voltage subgraph. 
            
        """
        # 1. Optimize parameters to find the ideal PDF
        _, pdf = self.optimizer.optimize(
            target_avg=avg_degree,
            max_deg=max_degree,
            dist_type=dist_type
        )
        
        # 2. Sample from the PDF
        degree_values = np.arange(1, max_degree + 1)
        samples = self.rng.choice(degree_values, size=n_nodes, p=pdf)
        return samples.tolist()

    def _generate_transformer_simple(self, n_nodes: int, prob: float) -> List[int]:
        """
        Original simple model: binary transformer degree list (0 or 1).
        """
        return self.rng.choice([0, 1], size=n_nodes, p=[1.0 - prob, prob]).tolist()

    def _generate_transformer_stars(self, n_i: int, n_j: int, 
                                    c: float = 0.174, gamma: float = 4.15) -> Tuple[List[int], List[int]]:
        """
        Generates transformer degrees based on the 'disjoint k-stars' model 
        from the paper.
        
        Args:
            n_i, n_j: Number of nodes in the two voltage levels.
            c: Coefficient for number of active stars (default 0.174).
            gamma: Power law exponent (default 4.15).
            
        Returns:
            Tuple (t_i, t_j) representing degree sequences for both levels.
        """
        # 1. Determine number of 'star centers' (active hubs)
        # h(n_i, n_j) = c * min(n_i, n_j)
        n_stars = int(max(1, np.round(c * min(n_i, n_j))))
        
        # 2. Generate Star Sizes from Power Law
        # Since gamma is high (4.15), most stars are small (size 1 or 2).
        # We sample from range [1, 100] (arbitrary cap, probability of >100 is negligible)
        max_star_size = 100
        x = np.arange(1, max_star_size + 1)
        probs = x ** (-gamma)
        probs /= probs.sum()
        
        star_sizes = self.rng.choice(x, size=n_stars, p=probs)
        
        # 3. Assign Stars to create Degree Sequences
        # To satisfy bipartite constraint (sum(d_i) == sum(d_j)), we build pairs.
        # For a star of size k:
        #   - Center in Level I: Level I gets one node deg=k, Level J gets k nodes deg=1
        #   - Center in Level J: Level J gets one node deg=k, Level I gets k nodes deg=1
        
        # We start with empty lists of non-zero degrees
        active_degrees_i = []
        active_degrees_j = []
        
        for size in star_sizes:
            # Randomly assign center to I or J (50/50 split assumption)
            if self.rng.random() < 0.5:
                # Center in I
                active_degrees_i.append(size)
                active_degrees_j.extend([1] * size)
            else:
                # Center in J
                active_degrees_j.append(size)
                active_degrees_i.extend([1] * size)
                
        # 4. Pad with zeros to match full node counts
        # We need to ensure we don't have more active nodes than total nodes
        if len(active_degrees_i) > n_i:
            # Truncate if we generated too many (rare given c=0.174)
            active_degrees_i = active_degrees_i[:n_i]
        else:
            active_degrees_i.extend([0] * (n_i - len(active_degrees_i)))
            
        if len(active_degrees_j) > n_j:
            active_degrees_j = active_degrees_j[:n_j]
        else:
            active_degrees_j.extend([0] * (n_j - len(active_degrees_j)))
            
        # 5. Shuffle to distribute connections randomly
        self.rng.shuffle(active_degrees_i)
        self.rng.shuffle(active_degrees_j)
        
        return active_degrees_i, active_degrees_j

    def create_params(self, 
                      levels: List[Dict[str, Any]], 
                      inter_connections: Dict[Tuple[int, int], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates the full parameter set.
        
        Args:
            levels: TODO: 
            inter_connections: Dict mapping (i, j) to config.
                               Config can be {'type': 'simple', 'p_i_j': ..., 'p_j_i': ...}
                               OR {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        
        TODO: We can further improve the setup for `max_d`. For `dgln`, the paper suggests $\bar{d}=2.425\pm 0.1846$ and $d_{\max}$ is rather difficult to generalize as a function of $n$. For `dpl`, some work suggests $d_\max\sim n^{1/\gamma}$ with $\gamma\in[1,4]$. Using $g(n)=c\cdot n^{1/4}$ to fit some data yields $c\approx 1.517$. 
        """
        degrees_by_level = []
        diameters_by_level = []
        
        # --- 1. Voltage Levels ---
        for i, lvl_cfg in enumerate(levels):
            n = int(lvl_cfg['n'])
            avg_k = float(lvl_cfg['avg_k'])
            d = int(lvl_cfg['diam'])
            dist_type = lvl_cfg.get('dist_type', 'poisson')
            max_k = lvl_cfg.get('max_k', min(n - 1, 50))

            print(f"Generating Level {i}: {dist_type.upper()} distribution (Avg={avg_k})")

            if dist_type == 'poisson':
                deg_seq = self._generate_poisson_degrees(n, avg_k)
            else:
                deg_seq = self._generate_optimized_degrees(n, avg_k, max_k, dist_type)
            
            degrees_by_level.append(deg_seq)
            diameters_by_level.append(d)

        # --- 2. Transformer Connections ---
        transformer_degrees = {}
        
        for (i, j), cfg in inter_connections.items():
            if i >= len(levels) or j >= len(levels): continue
                
            n_i = int(levels[i]['n'])
            n_j = int(levels[j]['n'])
            
            conn_type = cfg.get('type', 'simple')
            
            if conn_type == 'k-stars':
                print(f"Generating Transformers {i}<->{j}: k-Stars Model")
                c_val = cfg.get('c', 0.174)
                gamma_val = cfg.get('gamma', 4.15)
                print(gamma_val)
                t_i, t_j = self._generate_transformer_stars(n_i, n_j, c_val, gamma_val)
            else:
                # Simple probabilistic model
                print(f"Generating Transformers {i}<->{j}: Simple Probabilistic")
                p_i = cfg.get('p_i_j', 0.0)
                p_j = cfg.get('p_j_i', 0.0)
                t_i = self._generate_transformer_simple(n_i, p_i)
                t_j = self._generate_transformer_simple(n_j, p_j)
            
            transformer_degrees[(i, j)] = (t_i, t_j)

        return {
            "degrees_by_level": degrees_by_level,
            "diameters_by_level": diameters_by_level,
            "transformer_degrees": transformer_degrees
        }