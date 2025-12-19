import pytest
import numpy as np
import networkx as nx
from powergrid_synth.generation_dispatcher import GenerationDispatcher

@pytest.fixture
def sample_grid():
    G = nx.Graph()
    # 10 Generators, 100 MW each = 1000 MW total
    for i in range(10):
        G.add_node(i, bus_type='Gen', pg_max=100.0)
    
    # Loads
    G.add_node(10, bus_type='Load', pl=200.0)
    G.add_node(11, bus_type='Load', pl=200.0)
    # Total Load = 400 MW (40% Loading)
    return G

class TestGenerationDispatcher:

    def test_initialization(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        assert dispatcher.mu_committed is not None

    def test_select_uncommitted(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid)
        norm_pg = np.array([[i, 1.0] for i in range(10)])
        
        uncomm, remaining = dispatcher._select_uncommitted(norm_pg)
        assert 1 <= len(uncomm) <= 2
        assert uncomm.shape[1] == 3
        assert np.all(uncomm[:, 2] == 0.0)

    def test_select_committed(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid)
        norm_pg = np.array([[i, 1.0] for i in range(8)])
        
        # Pass total units count (10) explicitly
        comm, remaining = dispatcher._select_committed(norm_pg, 10)
        # 40-50% of 10 is 4-5. 
        # But random factors apply. Just check bounds.
        assert 3 <= len(comm) <= 6

    def test_dispatch_balancing_light(self, sample_grid):
        """Test dispatch with 400 MW load (Excess scenario initial)."""
        dispatcher = GenerationDispatcher(sample_grid)
        result = dispatcher.dispatch()
        
        total_gen = sum(result.values())
        total_load = 400.0
        
        print(f"Target: {total_load}, Actual: {total_gen}")
        assert abs(total_gen - total_load) < 0.1 * total_load

    def test_dispatch_balancing_heavy(self, sample_grid):
        """Test dispatch with 800 MW load (Deficit scenario initial)."""
        sample_grid.nodes[10]['pl'] = 400.0
        sample_grid.nodes[11]['pl'] = 400.0 
        
        dispatcher = GenerationDispatcher(sample_grid)
        result = dispatcher.dispatch()
        
        total_gen = sum(result.values())
        total_load = 800.0
        
        print(f"Target: {total_load}, Actual: {total_gen}")
        assert abs(total_gen - total_load) < 0.1 * total_load