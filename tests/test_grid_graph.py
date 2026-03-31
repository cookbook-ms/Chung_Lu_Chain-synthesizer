import pytest
import networkx as nx
from powergrid_synth.grid_graph import PowerGridGraph


class TestPowerGridGraph:

    @pytest.fixture()
    def multi_level_graph(self):
        g = PowerGridGraph()
        for i in range(5):
            g.add_node(i, voltage_level=0)
        for i in range(5, 15):
            g.add_node(i, voltage_level=1)
        for i in range(15, 25):
            g.add_node(i, voltage_level=2)
        # intra-level edges
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        g.add_edges_from([(5, 6), (6, 7), (7, 8)])
        # inter-level (transformer) edges
        g.add_edges_from([(4, 5), (8, 15)])
        return g

    def test_level_returns_correct_nodes(self, multi_level_graph):
        sub0 = multi_level_graph.level(0)
        assert set(sub0.nodes()) == {0, 1, 2, 3, 4}

    def test_level_returns_correct_nodes_level1(self, multi_level_graph):
        sub1 = multi_level_graph.level(1)
        assert set(sub1.nodes()) == set(range(5, 15))

    def test_level_contains_only_intra_edges(self, multi_level_graph):
        sub0 = multi_level_graph.level(0)
        for u, v in sub0.edges():
            assert sub0.nodes[u]['voltage_level'] == 0
            assert sub0.nodes[v]['voltage_level'] == 0

    def test_level_nonexistent_returns_empty(self, multi_level_graph):
        sub99 = multi_level_graph.level(99)
        assert sub99.number_of_nodes() == 0

    def test_is_subclass_of_nx_graph(self):
        g = PowerGridGraph()
        assert isinstance(g, nx.Graph)
