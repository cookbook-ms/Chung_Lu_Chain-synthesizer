import pytest
import networkx as nx
from unittest.mock import patch
import matplotlib.pyplot as plt

# Import your package modules
from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.bus_type_allocator import BusTypeAllocator
from powergrid_synth.visualization_old import GridVisualizer

class TestIntegration:
    
    @patch('matplotlib.pyplot.show') # Prevent plot window from blocking tests
    def test_full_workflow_execution(self, mock_show):
        """
        Runs a complete end-to-end test:
        1. Configure Input
        2. Generate Topology (3 levels)
        3. Allocat Bus Types
        4. Visualize (smoke test)
        """
        # 1. Configuration
        configurator = InputConfigurator(seed=123)
        level_specs = [
            {'n': 20, 'avg_k': 3.0, 'diam': 5, 'dist_type': 'poisson'},
            {'n': 40, 'avg_k': 2.0, 'diam': 8, 'dist_type': 'poisson'}
        ]
        connection_specs = {
            (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        }
        params = configurator.create_params(level_specs, connection_specs)
        
        # 2. Generation
        gen = PowerGridGenerator(seed=123)
        grid = gen.generate_grid(
            params['degrees_by_level'], 
            params['diameters_by_level'], 
            params['transformer_degrees'],
            keep_lcc=True
        )
        
        assert grid.number_of_nodes() > 0
        assert grid.number_of_edges() > 0
        
        # 3. Bus Type Allocation
        allocator = BusTypeAllocator(grid)
        # Use small max_iter for speed in testing
        bus_types = allocator.allocate(max_iter=50, population_size=5)
        
        nx.set_node_attributes(grid, bus_types, name="bus_type")
        
        # Check that attributes are actually set
        node_sample = list(grid.nodes())[0]
        assert 'bus_type' in grid.nodes[node_sample]
        assert grid.nodes[node_sample]['bus_type'] in ['Gen', 'Load', 'Conn']

        # 4. Visualization (Smoke Test)
        # We call the plotting functions to ensure they don't crash.
        # The @patch decorator prevents the window from actually opening.
        viz = GridVisualizer()
        viz.plot_grid(grid, layout='spring', title="Integration Test Grid")
        viz.plot_bus_types(grid, layout='spring', title="Integration Test Bus Types")
        
        # Also test the interactive helper methods (just ensuring no crash)
        # Note: We can't fully test interactivity in CI, but we can call the setup logic
        try:
            viz.plot_interactive(grid)
            viz.plot_interactive_bus_types(grid)
        except Exception as e:
            pytest.fail(f"Interactive plot setup failed: {e}")