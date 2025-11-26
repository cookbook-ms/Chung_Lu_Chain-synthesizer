import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Button
from typing import Dict, Any, Tuple, List, Optional

class MatplotlibDropdown:
    """
    A custom Dropdown widget for Matplotlib, created by combining Buttons.
    """
    def __init__(self, fig, rect, labels, active=0, on_select=None):
        self.fig = fig
        self.labels = labels
        self.active_idx = active
        self.on_select = on_select
        self.is_menu_open = False
        self.buttons = []
        
        # Dimensions
        left, bottom, width, height = rect
        self.rect = rect
        self.height = height
        
        # 1. Create the Main Button (Header)
        # Position: The top of the defined rect
        ax_main = plt.axes([left, bottom, width, height])
        self.main_btn = Button(ax_main, labels[active], color='#e0e0e0', hovercolor='#d0d0d0')
        self.main_btn.on_clicked(self.toggle_menu)
        
        # 2. Create Option Buttons (Hidden by default)
        # They will appear BELOW the main button
        self.option_axes = []
        self.option_btns = []
        
        for i, label in enumerate(labels):
            # Calculate position: stacked downwards
            # y = bottom - (i+1)*height
            y_pos = bottom - ((i + 1) * height)
            ax_opt = plt.axes([left, y_pos, width, height])
            ax_opt.set_visible(False) # Hide initially
            
            btn = Button(ax_opt, label, color='white', hovercolor='#f0f0f0')
            # We need to bind the specific index/label to the callback
            # Using a default argument in lambda captures the current value of i/label
            btn.on_clicked(lambda event, idx=i, lbl=label: self.select_option(idx, lbl))
            
            self.option_axes.append(ax_opt)
            self.option_btns.append(btn)

    def toggle_menu(self, event):
        self.is_menu_open = not self.is_menu_open
        for ax in self.option_axes:
            ax.set_visible(self.is_menu_open)
        self.fig.canvas.draw_idle()

    def select_option(self, idx, label):
        # Update state
        self.active_idx = idx
        self.main_btn.label.set_text(label)
        
        # Close menu
        self.is_menu_open = False
        for ax in self.option_axes:
            ax.set_visible(False)
            
        self.fig.canvas.draw_idle()
        
        # Trigger callback
        if self.on_select:
            self.on_select(label)

class GridVisualizer:
    """
    Visualization module for synthetic power grids.
    Allows plotting the grid with different layouts including Yifan Hu, Kamada-Kawai, and Voltage Layered.
    """

    def __init__(self):
        # Default colormap for voltage levels
        self.cmap = cm.get_cmap('tab10') 
        # Keep references to widgets to prevent garbage collection
        self._widgets = []

    def _get_node_colors(self, graph: nx.Graph) -> List[Any]:
        """Assigns colors to nodes based on their 'voltage_level' attribute."""
        colors = []
        for node in graph.nodes():
            level = graph.nodes[node].get('voltage_level', 0)
            colors.append(self.cmap(level))
        return colors

    def _get_layered_layout(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Custom layout: Places nodes in horizontal bands based on voltage level."""
        pos = {}
        levels = {}
        for node, data in graph.nodes(data=True):
            lvl = data.get('voltage_level', 0)
            if lvl not in levels:
                levels[lvl] = []
            levels[lvl].append(node)
        
        max_width = max([len(nodes) for nodes in levels.values()]) if levels else 1
        
        for lvl, nodes in levels.items():
            y = -lvl * 10 
            x_values = np.linspace(-max_width/2, max_width/2, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = np.array([x_values[i], y])
        return pos

    def _yifan_hu_layout(self, G: nx.Graph, iterations: int = 100, k: Optional[float] = None) -> Dict[int, np.ndarray]:
        """Implementation of the Yifan Hu force-directed layout algorithm."""
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0: return {}
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
        pos = np.random.rand(n, 2) * 2 - 1
        
        if k is None:
            k = 1.0 / np.sqrt(n) if n > 0 else 1.0
        
        step = n / 10.0
        
        for it in range(iterations):
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(delta**2, axis=2)
            dist = np.sqrt(dist_sq)
            dist[dist == 0] = 0.001
            
            fr = (k**2 / dist_sq[..., np.newaxis]) * delta
            np.fill_diagonal(fr[:, :, 0], 0)
            np.fill_diagonal(fr[:, :, 1], 0)
            
            displacement = np.sum(fr, axis=1)
            
            rows, cols = adj_matrix.nonzero()
            for u, v in zip(rows, cols):
                if u >= v: continue
                delta_uv = pos[v] - pos[u]
                dist_uv = np.linalg.norm(delta_uv)
                if dist_uv == 0: continue
                fa_mag = dist_uv / k
                fa_vec = fa_mag * (delta_uv / dist_uv)
                displacement[u] += fa_vec
                displacement[v] -= fa_vec

            length = np.linalg.norm(displacement, axis=1)
            length[length == 0] = 0.1
            scale = np.minimum(step, length) / length
            pos += displacement * scale[:, np.newaxis]
            step *= 0.95 

        return {nodes[i]: pos[i] for i in range(n)}

    def plot_grid(self, graph: nx.Graph, layout: str = 'kamada_kawai', title: str = "Grid", show_labels: bool = False, figsize: Tuple[int, int] = (12, 10)):
        """Static plot function."""
        plt.figure(figsize=figsize)
        self._draw_graph_on_ax(plt.gca(), graph, layout, title, show_labels)
        plt.tight_layout()
        plt.show()

    def _draw_graph_on_ax(self, ax, graph, layout_name, title, show_labels):
        """Helper to draw graph on a specific axis."""
        print(f"Calculating layout '{layout_name}'...")
        if layout_name == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout_name == 'spring':
            pos = nx.spring_layout(graph, seed=42)
        elif layout_name == 'spectral':
            pos = nx.spectral_layout(graph)
        elif layout_name == 'voltage_layered':
            pos = self._get_layered_layout(graph)
        elif layout_name == 'yifan_hu':
            pos = self._yifan_hu_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        node_colors = self._get_node_colors(graph)
        
        ax.clear()
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=50, alpha=0.9)
        
        if show_labels:
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
            
        ax.set_title(f"{title}\nLayout: {layout_name}")
        ax.axis('off')

        # Add legend
        unique_levels = sorted(list(set(nx.get_node_attributes(graph, 'voltage_level').values())))
        import matplotlib.patches as mpatches
        legend_elements = [mpatches.Patch(color=self.cmap(lvl), label=f'Voltage Level {lvl}') for lvl in unique_levels]
        ax.legend(handles=legend_elements, loc='upper right')

    def _create_interactive_window(self, graph: nx.Graph, title: str, figsize: Tuple[int, int]):
        """Helper to create a figure with a Dropdown menu for layout selection."""
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(top=0.85)  # Reserve space at top for dropdown
        
        # Define Dropdown Position (Top Left corner)
        # Rect: [left, bottom, width, height]
        # We put it near the top left
        dropdown_rect = [0.05, 0.9, 0.20, 0.05] 
        
        layout_options = ['kamada_kawai', 'yifan_hu', 'spring', 'spectral', 'voltage_layered']
        
        # Callback wrapper
        def update_layout(label):
            self._draw_graph_on_ax(ax, graph, label, title, show_labels=False)
            fig.canvas.draw_idle()

        # Create our custom dropdown
        dropdown = MatplotlibDropdown(fig, dropdown_rect, layout_options, active=0, on_select=update_layout)
        
        # Label next to dropdown
        fig.text(0.05, 0.96, "Select Layout:", weight='bold')

        # Keep reference to prevent GC
        self._widgets.append(dropdown)
        
        # Initial draw
        update_layout(layout_options[0])
        plt.show()

    def plot_interactive(self, graph: nx.Graph, title: str = "Interactive Grid", figsize: Tuple[int, int] = (14, 10)):
        """Opens an interactive window for the full grid."""
        self._create_interactive_window(graph, title, figsize)

    def plot_interactive_voltage_level(self, graph: nx.Graph, level: int, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """Opens an interactive window for a specific voltage level."""
        nodes = [n for n, d in graph.nodes(data=True) if d.get('voltage_level') == level]
        if not nodes:
            print(f"No nodes found for voltage level {level}")
            return
            
        subgraph = graph.subgraph(nodes)
        if title is None:
            title = f"Voltage Level {level} (Interactive)"
        
        self._create_interactive_window(subgraph, title, figsize)