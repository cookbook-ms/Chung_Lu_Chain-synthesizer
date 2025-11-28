import pytest
import numpy as np
from powergrid_synth.input_configurator import InputConfigurator

class TestInputConfig:
    def setup_method(self):
        self.config = InputConfigurator(seed=42)

    def test_poisson_generation(self):
        """Test basic degree list generation."""
        n = 100
        avg = 3.0
        degrees = self.config._generate_poisson_degrees(n, avg)
        
        assert len(degrees) == n
        assert abs(np.mean(degrees) - avg) < 1.0 # Rough statistical check

    def test_k_stars_generation(self):
        """Test the k-stars transformer logic."""
        n_i = 50
        n_j = 100
        c = 0.174
        gamma = 4.15
        
        t_i, t_j = self.config._generate_transformer_stars(n_i, n_j, c, gamma)
        
        assert len(t_i) == n_i
        assert len(t_j) == n_j
        
        # Bipartite constraint check:
        # Sum of degrees leaving I must equal sum of degrees leaving J
        assert sum(t_i) == sum(t_j)

    def test_create_params_structure(self):
        """Test the full dictionary creation."""
        levels = [{'n': 10, 'avg_k': 2, 'diam': 3}]
        conns = {}
        
        params = self.config.create_params(levels, conns)
        
        assert 'degrees_by_level' in params
        assert 'diameters_by_level' in params
        assert 'transformer_degrees' in params
        assert len(params['degrees_by_level']) == 1