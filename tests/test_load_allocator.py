import pytest
import networkx as nx
import numpy as np
from powergrid_synth.load_allocator import LoadAllocator

@pytest.fixture
def sample_graph():
    """Creates a small mock graph with Gen, Load, and Conn buses."""
    G = nx.Graph()
    # Create Gen nodes with pg_max attribute
    G.add_node(0, bus_type='Gen', pg_max=100.0)
    G.add_node(1, bus_type='Gen', pg_max=50.0)
    
    # Create Load nodes
    G.add_node(2, bus_type='Load')
    G.add_node(3, bus_type='Load')
    G.add_node(4, bus_type='Load')
    
    # Create Conn nodes
    G.add_node(5, bus_type='Conn')
    
    # Add edges (degrees matter for allocation)
    # Node 2 deg 2, Node 3 deg 2, Node 4 deg 2
    G.add_edges_from([(0, 2), (0, 3), (1, 4), (2, 3), (3, 4), (4, 5)])
    return G

class TestLoadAllocator:
    
    def test_initialization(self, sample_graph):
        allocator = LoadAllocator(sample_graph)
        # Should identify 3 load buses (2, 3, 4)
        assert len(allocator.load_buses) == 3
        assert set(allocator.load_buses) == {2, 3, 4}

    def test_total_load_calculation_deterministic(self, sample_graph):
        allocator = LoadAllocator(sample_graph)
        # N=6. Formula check or just ensure it returns positive float
        total_load = allocator._calculate_total_load('D', 150.0)
        assert total_load > 0
        assert isinstance(total_load, float)

    def test_total_load_calculation_levels(self, sample_graph):
        allocator = LoadAllocator(sample_graph)
        total_gen = 150.0
        
        # Heavy: 0.7 to 0.8 * Gen
        load_h = allocator._calculate_total_load('H', total_gen)
        assert 0.7 * total_gen <= load_h <= 0.8 * total_gen
        
        # Medium: 0.5 to 0.6 * Gen
        load_m = allocator._calculate_total_load('M', total_gen)
        assert 0.5 * total_gen <= load_m <= 0.6 * total_gen
        
        # Light: 0.3 to 0.4 * Gen
        load_l = allocator._calculate_total_load('L', total_gen)
        assert 0.3 * total_gen <= load_l <= 0.4 * total_gen

    def test_allocation_execution(self, sample_graph):
        # Use Reference System 1
        allocator = LoadAllocator(sample_graph, ref_sys_id=1)
        loads = allocator.allocate(loading_level='H')
        
        assert len(loads) == 3
        assert all(isinstance(val, float) for val in loads.values())
        assert all(val > 0 for val in loads.values())
        
        # Check consistency
        total_assigned = sum(loads.values())
        assert total_assigned > 0

    def test_heuristic_fallback(self, sample_graph):
        # Test with ref_sys_id=0 (Heuristic)
        allocator = LoadAllocator(sample_graph, ref_sys_id=0)
        loads = allocator.allocate(loading_level='M')
        assert len(loads) == 3
        assert sum(loads.values()) > 0

    def test_no_load_buses(self):
        # Graph with no loads
        G = nx.Graph()
        G.add_node(0, bus_type='Gen', pg_max=100)
        allocator = LoadAllocator(G)
        loads = allocator.allocate()
        assert loads == {}

    def test_zero_generation_fallback(self, sample_graph):
        # If generation is 0, 'H' loading might fail if not handled, 
        # but the code switches to 'D' (Deterministic) if Gen=0.
        G = nx.Graph()
        G.add_node(0, bus_type='Load') # No Gens
        allocator = LoadAllocator(G)
        
        # Should rely on 'D' formula based on N=1
        loads = allocator.allocate(loading_level='H')
        assert len(loads) == 1
        assert list(loads.values())[0] > 0