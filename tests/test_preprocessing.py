import pytest
import numpy as np
from powergrid_synth.preprocessing import Preprocessor

class TestPreprocessor:
    def setup_method(self):
        self.prep = Preprocessor()

    def test_degree_inflation(self):
        """Test that degree inflation works and doesn't return fewer nodes."""
        initial_degrees = [3, 3, 2, 1, 1] # 5 nodes
        target_diameter = 4
        
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # Check that we didn't lose nodes
        assert len(d_prime) >= len(initial_degrees)
        
        # Check that we have a numpy array
        assert isinstance(d_prime, np.ndarray)

    def test_diameter_path_constraints(self):
        """Test that the Diameter Path (D) nodes are assigned to distinct boxes."""
        initial_degrees = [4]*20 # plenty of high degree nodes
        target_diameter = 5
        
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # Diameter set size should be diameter + 1
        assert len(D) == target_diameter + 1
        
        # Check distinct box assignment
        # v[i] is the box ID for node i
        boxes_for_D = [v[i] for i in D]
        assert len(boxes_for_D) == len(set(boxes_for_D)), "Nodes in D must have distinct boxes"

    def test_subdiameter_path_constraints(self):
        """Test Subdiameter Path (S) logic."""
        initial_degrees = [4]*50
        target_diameter = 10
        
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # S should be disjoint from D
        assert len(D.intersection(S)) == 0
        
        if len(S) > 0:
            boxes_for_S = [v[i] for i in S]
            assert len(boxes_for_S) == len(set(boxes_for_S)), "Nodes in S must have distinct boxes"