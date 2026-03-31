import pytest
import numpy as np
import networkx as nx
from powergrid_synth.transmission import TransmissionLineAllocator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.bus_type_allocator import BusTypeAllocator
from powergrid_synth.capacity_allocator import CapacityAllocator
from powergrid_synth.load_allocator import LoadAllocator
from powergrid_synth.generation_dispatcher import GenerationDispatcher


class TestTransmissionLineAllocator:

    @pytest.fixture()
    def dispatched_grid(self):
        """Small grid with bus types, capacities, loads, and dispatch set."""
        np.random.seed(42)
        configurator = InputConfigurator(seed=42)
        level_specs = [
            {"n": 25, "avg_k": 2.5, "diam": 5, "dist_type": "poisson"},
        ]
        params = configurator.create_params(level_specs, {})

        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            params["degrees_by_level"],
            params["diameters_by_level"],
            params["transformer_degrees"],
            keep_lcc=True,
        )

        bus_types = BusTypeAllocator(grid).allocate(max_iter=5)
        nx.set_node_attributes(grid, bus_types, name="bus_type")

        caps = CapacityAllocator(grid, ref_sys_id=1).allocate()
        nx.set_node_attributes(grid, caps, name="pg_max")

        loads = LoadAllocator(grid, ref_sys_id=1).allocate(loading_level="M")
        nx.set_node_attributes(grid, loads, name="pl")

        dispatch = GenerationDispatcher(grid, ref_sys_id=1).dispatch()
        nx.set_node_attributes(grid, dispatch, name="pg")

        return grid

    def test_allocate_returns_dict(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        assert isinstance(result, dict)

    def test_all_edges_get_capacity(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        assert len(result) == dispatched_grid.number_of_edges()

    def test_edges_have_impedance_attrs(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        allocator.allocate()
        for u, v, data in dispatched_grid.edges(data=True):
            assert "x" in data, f"Edge ({u},{v}) missing 'x' attribute"
            assert "r" in data, f"Edge ({u},{v}) missing 'r' attribute"
            assert data["x"] > 0

    def test_generate_phi_range(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        phi = allocator._generate_phi(100)
        assert phi.shape == (100,)
        assert np.all(phi >= 0.01)
        assert np.all(phi <= 89.99)

    def test_generate_beta(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        beta = allocator._generate_beta(50)
        assert beta.shape == (50,)
        assert np.all(beta > 0)

    def test_capacities_are_positive(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        for cap in result.values():
            assert cap > 0
