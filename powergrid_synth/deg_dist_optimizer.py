import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional

class DegreeDistributionOptimizer:
    """
    Finds parameters for 'ideal' degree distributions (DGLN or Power Law)
    matching a target average degree and maximum degree bound.
    
    Ported from MATLAB 'degdist_param_search.m' (Sandia National Labs).
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _dgln_pdf(self, n: int, alpha: float, beta: float) -> np.ndarray:
        """
        Discrete Generalized Log-Normal PDF.
        Prob(x) ~ exp(-(log(x)/alpha)^beta) for x = 1..n
        """
        # Avoid division by zero or invalid log domain issues during optimization
        if alpha <= 0 or beta <= 0:
            return np.zeros(n)
            
        x = np.arange(1, n + 1)
        # Calculate unnormalized probs
        # We use np.abs to handle potential negative params proposed by optimizer before bounds check
        try:
            p = np.exp(-((np.log(x)) / alpha) ** beta)
        except RuntimeWarning:
             return np.zeros(n)

        # Normalize
        total = np.sum(p)
        if total > 0:
            p = p / total
        else:
            # Fallback for bad parameters
            p = np.zeros(n)
            p[0] = 1.0 
            
        return p

    def _dpl_pdf(self, n: int, gamma: float) -> np.ndarray:
        """
        Discrete Power Law PDF.
        Prob(x) ~ x^-gamma for x = 1..n
        """
        x = np.arange(1, n + 1)
        p = x ** (-gamma)
        p = p / np.sum(p)
        return p

    def _objective_func(self, params, dist_type: str, max_deg: int, 
                        target_avg: float, prob_bound: float) -> float:
        """
        Calculates the score (penalty) for a set of parameters.
        Score = (Penalty if P(max_deg) > bound) + (Squared Error of Avg Degree)
        """
        if dist_type == 'dgln':
            alpha, beta = params
            # Hard constraint penalties for solver
            if alpha <= 0 or beta <= 0: return 1e6
            p = self._dgln_pdf(max_deg, alpha, beta)
        elif dist_type == 'dpl':
            gamma = params[0]
            if gamma <= 0: return 1e6
            p = self._dpl_pdf(max_deg, gamma)
        else:
            raise ValueError(f"Unknown type {dist_type}")

        # 1. Check Max Degree Bound Penalty
        # We want P(max_deg) < prob_bound
        p_max = p[-1]
        if p_max > prob_bound:
            # Penalty grows exponentially if bound is violated
            y1 = (np.exp(1 + p_max - prob_bound))**2 - 1
        else:
            y1 = 0.0

        # 2. Check Average Degree Error
        # Expected average degree = sum(x * p(x))
        x = np.arange(1, max_deg + 1)
        current_avg = np.sum(x * p)
        y2 = (current_avg - target_avg)**2

        score = y1 + y2
        
        if self.verbose:
            print(f"Params: {params}, Score: {score:.6f}, Avg: {current_avg:.4f}")
            
        return score

    def optimize(self, 
                 target_avg: float, 
                 max_deg: int, 
                 dist_type: str = 'dgln', 
                 prob_bound: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main entry point to find parameters.
        
        Returns:
            Tuple containing:
            - Best Parameters (alpha, beta) or (gamma,)
            - The resulting Probability Density Function (PDF) array
        """
        if dist_type == 'dgln':
            initial_guess = [2.0, 2.0]
        elif dist_type == 'dpl':
            initial_guess = [2.0]
        else:
            raise ValueError("Type must be 'dgln' or 'dpl'")

        # Run optimization (Nelder-Mead is typically robust for this)
        result = minimize(
            self._objective_func,
            initial_guess,
            args=(dist_type, max_deg, target_avg, prob_bound),
            method='Nelder-Mead',
            options={'xatol': 1e-4, 'fatol': 1e-4}
        )

        best_params = result.x
        
        # Generate final PDF
        if dist_type == 'dgln':
            final_pdf = self._dgln_pdf(max_deg, best_params[0], best_params[1])
        else:
            final_pdf = self._dpl_pdf(max_deg, best_params[0])
            
        if self.verbose:
            print(f"Optimization Success: {result.success}")
            print(f"Found Params: {best_params}")
            
        return best_params, final_pdf