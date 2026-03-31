import pytest
import numpy as np
import networkx as nx
from powergrid_synth.capacity_allocator import CapacityAllocator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.bus_type_allocator import BusTypeAllocator


class TestCapacityAllocator:

    @pytest.fixture()
    def grid_with_bus_types(self):
        np.random.seed(42)
        configurator = InputConfigurator(seed=42)
        level_specs = [
            {"n": 30, "avg_k": 2.5, "diam": 6, "dist_type": "poisson"},
        ]
        params = configurator.create_params(level_specs, {})
        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            params["degrees_by_level"],
            params["diameters_by_level"],
            params["transformer_degrees"],
            keep_lcc=True,
        )
        bus_types = BusTypeAllocator(grid).allocate(max_iter=10)
        nx.set_node_attributes(grid, bus_types, name="bus_type")
        return grid

    def test_allocate_returns_dict(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        assert isinstance(result, dict)

    def test_only_gen_nodes_get_capacity(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        gen_nodes = {
            n for n, d in grid_with_bus_types.nodes(data=True)
            if d.get("bus_type") == "Gen"
        }
        assert set(result.keys()) == gen_nodes

    def test_capacities_are_positive(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        for cap in result.values():
            assert cap > 0

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_different_ref_systems(self, grid_with_bus_types, ref_sys_id):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=ref_sys_id)
        result = allocator.allocate()
        assert len(result) > 0
