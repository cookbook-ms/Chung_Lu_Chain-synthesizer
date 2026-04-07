import pytest
import networkx as nx

pp = pytest.importorskip("pandapower", reason="pandapower not installed")
pytest.importorskip("pypowsybl", reason="pypowsybl not installed")

from powergrid_synth.data_format_converter import (
    pandapower_to_nx,
    nx_to_pandapower,
    pandapower_to_pypowsybl,
)


class TestPandapowerToNx:

    def test_case9_conversion(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(net.bus)

    def test_nodes_have_voltage_level(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        for _, data in G.nodes(data=True):
            assert "voltage_level" in data

    def test_nodes_have_bus_type(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        bus_types = {data["bus_type"] for _, data in G.nodes(data=True)}
        assert bus_types.issubset({"Gen", "Load", "Conn"})

    def test_edges_present(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert G.number_of_edges() > 0

    def test_base_kv_map_stored(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert "base_kv_map" in G.graph

    def test_case118_conversion(self):
        net = pp.networks.case118()
        G = pandapower_to_nx(net)
        assert G.number_of_nodes() == len(net.bus)
        assert G.number_of_edges() >= len(net.line)


class TestNxToPandapower:

    def test_roundtrip_case9(self):
        """Convert case9 to nx and back to pandapower."""
        net_orig = pp.networks.case9()
        G = pandapower_to_nx(net_orig)
        net_back = nx_to_pandapower(G)
        assert isinstance(net_back, pp.pandapowerNet)
        assert len(net_back.bus) == G.number_of_nodes()

    def test_from_fully_param_grid(self, fully_parameterised_grid):
        net = nx_to_pandapower(fully_parameterised_grid)
        assert isinstance(net, pp.pandapowerNet)
        assert len(net.bus) == fully_parameterised_grid.number_of_nodes()
        # Should have some generators
        assert len(net.gen) + len(net.ext_grid) > 0
        # Should have some loads
        assert len(net.load) > 0
        # Should have lines
        assert len(net.line) + len(net.trafo) > 0

    def test_custom_base_kv_map(self, fully_parameterised_grid):
        net = nx_to_pandapower(
            fully_parameterised_grid,
            base_mva=100.0,
            base_kv_map={0: 380.0, 1: 110.0},
        )
        assert isinstance(net, pp.pandapowerNet)


class TestPandapowerToPypowsybl:

    def test_convert_case9(self):
        net = pp.networks.case9()
        pp.runpp(net)
        ppl_net = pandapower_to_pypowsybl(net)
        assert ppl_net is not None
