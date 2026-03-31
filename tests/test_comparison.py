import pytest
import networkx as nx
import matplotlib
matplotlib.use("Agg")
from powergrid_synth.comparison import GraphComparator


class TestGraphComparator:

    @pytest.fixture()
    def two_graphs(self):
        g1 = nx.barabasi_albert_graph(40, 2, seed=10)
        g2 = nx.barabasi_albert_graph(45, 2, seed=20)
        return g1, g2

    @pytest.fixture()
    def level_graphs(self):
        """Two graphs with voltage_level attributes for per-level comparison."""
        def _make(n, seed):
            G = nx.barabasi_albert_graph(n, 2, seed=seed)
            for node in G.nodes():
                G.nodes[node]["voltage_level"] = 0 if node < n // 2 else 1
            return G
        return _make(40, 10), _make(45, 20)

    def test_init(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        assert comp.synth_graph is g1
        assert comp.ref_graph is g2

    def test_get_comparison_data_returns_dataframe(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        df = comp._get_comparison_data(g1, g2)
        assert "Metric" in df.columns
        assert len(df) == 8  # 8 metrics

    def test_print_metric_comparison(self, two_graphs, capsys):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.print_metric_comparison()
        captured = capsys.readouterr()
        assert "GRAPH COMPARISON REPORT" in captured.out

    def test_plot_degree_comparison(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_degree_comparison()  # should not raise

    def test_print_level_metrics(self, level_graphs, capsys):
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.print_level_metrics()
        captured = capsys.readouterr()
        assert "Level" in captured.out or "LEVEL" in captured.out or len(captured.out) > 0
