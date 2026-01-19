r"""
This module creates a PowerGridGraph class, inherited from nx.Graph, with some customized methods like accessing subgraph by specifying voltage_level.  
"""
import numpy as np
import networkx as nx

class PowerGridGraph(nx.Graph):
    """
    Custom NetworkX Graph for Power Grids.
    Extends nx.Graph to support subgraph extraction by voltage level.
    """
    def level(self, voltage_level: int) -> nx.Graph:
        """
        Returns a subgraph view containing only nodes at the specified voltage level.
        
        Args:
            voltage_level (int): The voltage level index (e.g., 0, 1).
            
        Returns:
            nx.Graph: A subgraph view of the grid.
        """
        nodes = [n for n, d in self.nodes(data=True) if d.get('voltage_level') == voltage_level]
        return self.subgraph(nodes)