import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Optional
from .analysis import GridAnalyzer

class GraphComparator:
    """
    Compares a synthetic power grid against a reference (real-world) graph.
    Provides tabular metric comparisons and visual distribution overlaps,
    both globally and per voltage level.
    """
    def __init__(self, synth_graph: nx.Graph, ref_graph: nx.Graph, 
                 synth_label: str = "Synthetic", ref_label: str = "Reference (Real)"):
        self.synth_graph = synth_graph
        self.ref_graph = ref_graph
        self.synth_label = synth_label
        self.ref_label = ref_label

    def _get_comparison_data(self, graph1: nx.Graph, graph2: nx.Graph) -> pd.DataFrame:
        """Helper to generate the dataframe for two specific graphs."""
        analyzer1 = GridAnalyzer(graph1)
        analyzer2 = GridAnalyzer(graph2)
        
        # Gather metrics
        s_basic = analyzer1.get_basic_stats()
        s_path = analyzer1.get_path_metrics()
        s_clust = analyzer1.get_clustering_metrics()
        
        r_basic = analyzer2.get_basic_stats()
        r_path = analyzer2.get_path_metrics()
        r_clust = analyzer2.get_clustering_metrics()

        data = {
            "Metric": [
                "Nodes", "Edges", "Density", 
                "Connected?", "Diameter (LCC)", "Avg Path Len (LCC)", 
                "Avg Clustering", "Transitivity"
            ],
            self.synth_label: [
                s_basic['num_nodes'], s_basic['num_edges'], f"{s_basic['density']:.6f}",
                "Yes" if s_path['is_connected'] else "No", 
                s_path['diameter'], f"{s_path['avg_path_length']:.4f}",
                f"{s_clust['avg_clustering_coef']:.4f}", f"{s_clust['transitivity']:.4f}"
            ],
            self.ref_label: [
                r_basic['num_nodes'], r_basic['num_edges'], f"{r_basic['density']:.6f}",
                "Yes" if r_path['is_connected'] else "No", 
                r_path['diameter'], f"{r_path['avg_path_length']:.4f}",
                f"{r_clust['avg_clustering_coef']:.4f}", f"{r_clust['transitivity']:.4f}"
            ]
        }
        return pd.DataFrame(data)

    def print_metric_comparison(self, synth_graph: Optional[nx.Graph] = None, 
                              ref_graph: Optional[nx.Graph] = None, 
                              title: str = "GRAPH COMPARISON REPORT"):
        """Prints a side-by-side table of topological metrics."""
        s_g = synth_graph if synth_graph is not None else self.synth_graph
        r_g = ref_graph if ref_graph is not None else self.ref_graph
        
        df = self._get_comparison_data(s_g, r_g)
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        print(df.to_string(index=False))
        print("="*60 + "\n")

    def plot_degree_comparison(self, synth_graph: Optional[nx.Graph] = None, 
                             ref_graph: Optional[nx.Graph] = None, 
                             ax: Optional[plt.Axes] = None,
                             log_scale: bool = True,
                             title: str = "Degree Distribution Comparison"):
        """
        Plots overlaid degree distributions.
        
        Args:
            synth_graph: Custom synthetic graph (or None for self.synth_graph).
            ref_graph: Custom reference graph (or None for self.ref_graph).
            ax: Matplotlib axis to plot on. If None, creates new figure.
            log_scale: Whether to use log-log scale (default True).
            title: Title for the plot.
        """
        s_g = synth_graph if synth_graph is not None else self.synth_graph
        r_g = ref_graph if ref_graph is not None else self.ref_graph
        
        deg_synth = [d for n, d in s_g.degree()]
        deg_ref = [d for n, d in r_g.degree()]
        
        if not deg_synth or not deg_ref:
            # Handle empty graphs
            return
        
        created_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_figure = True
        
        if log_scale:
            # Log-Log Plot
            # Helper to get log-log coordinates
            def get_log_coords(degrees):
                counts = np.bincount(degrees)
                vals = np.nonzero(counts)[0]
                return vals, counts[vals]
            
            x_s, y_s = get_log_coords(deg_synth)
            x_r, y_r = get_log_coords(deg_ref)
            
            ax.loglog(x_s, y_s, 'bo', markersize=5, alpha=0.6, label=self.synth_label)
            ax.loglog(x_r, y_r, 'r^', markersize=5, alpha=0.6, label=self.ref_label)
            
            ax.set_xlabel("Degree (log)")
            ax.set_ylabel("Count (log)")
        else:
            # Linear Histogram
            max_deg = max(max(deg_synth), max(deg_ref))
            bins = range(0, max_deg + 2)
            
            ax.hist(deg_synth, bins=bins, density=True, alpha=0.5, label=self.synth_label, color='blue')
            ax.hist(deg_ref, bins=bins, density=True, alpha=0.5, label=self.ref_label, color='orange')
            
            ax.set_xlabel("Degree")
            ax.set_ylabel("Probability Density")

        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        if created_figure:
            plt.tight_layout()
            plt.show()

    def print_level_metrics(self):
        """
        Iterates through voltage levels found in both graphs and prints metrics.
        Does NOT plot.
        """
        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        
        common_levels = sorted(list(levels_s.intersection(levels_r)))
        
        if not common_levels:
            print("No common 'voltage_level' attributes found between graphs.")
            return

        for level in common_levels:
            # Extract Subgraphs
            nodes_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            nodes_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]
            
            sub_s = self.synth_graph.subgraph(nodes_s)
            sub_r = self.ref_graph.subgraph(nodes_r)
            
            # Print Comparison
            self.print_metric_comparison(sub_s, sub_r, title=f"LEVEL {level} COMPARISON")

    def plot_all_levels_comparison(self, log_scale: bool = True):
        """
        Plots degree comparison for all common voltage levels in a single figure.
        """
        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        common_levels = sorted(list(levels_s.intersection(levels_r)))
        
        if not common_levels:
            print("No common levels to plot.")
            return

        n_levels = len(common_levels)
        cols = 3 if n_levels > 3 else n_levels
        rows = math.ceil(n_levels / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        if n_levels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        print(f"Plotting Combined Comparison Figure for {n_levels} Levels (Log Scale: {log_scale})...")
        
        for i, level in enumerate(common_levels):
            nodes_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            nodes_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]
            
            sub_s = self.synth_graph.subgraph(nodes_s)
            sub_r = self.ref_graph.subgraph(nodes_r)
            
            self.plot_degree_comparison(sub_s, sub_r, ax=axes[i], log_scale=log_scale, title=f"Level {level}")
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def run_full_comparison(self, log_scale: bool = True):
        """Runs global comparison followed by per-level comparison."""
        print(">>> Running Global Comparison")
        self.print_metric_comparison(title="GLOBAL GRAPH COMPARISON")
        self.plot_degree_comparison(title="Global Degree Comparison", log_scale=log_scale)
        
        print(">>> Running Per-Level Metric Comparison")
        self.print_level_metrics()
        
        print(">>> Plotting Per-Level Comparisons")
        self.plot_all_levels_comparison(log_scale=log_scale)