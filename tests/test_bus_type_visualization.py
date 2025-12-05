import sys
import os
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.bus_type_allocator import BusTypeAllocator
from powergrid_synth.visualization import GridVisualizer

def main():
    print("--- 1. Generating 3-Level Grid ---")
    configurator = InputConfigurator(seed=42)
    
    # We use a smaller grid for clearer visualization of edge styles
    level_specs = [
        {'n': 30, 'avg_k': 3.5, 'diam': 6, 'dist_type': 'dgln'},
        {'n': 60, 'avg_k': 2.5, 'diam': 10, 'dist_type': 'dpl'},
        {'n': 120, 'avg_k': 2.0, 'diam': 15, 'dist_type': 'poisson'}
    ]
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }
    
    params = configurator.create_params(level_specs, connection_specs)
    gen = PowerGridGenerator(seed=42)
    grid = gen.generate_grid(
        params['degrees_by_level'], 
        params['diameters_by_level'], 
        params['transformer_degrees'],
        keep_lcc=True
    )
    
    print(f"Grid Size: {grid.number_of_nodes()} nodes, {grid.number_of_edges()} edges")
    
    print("\n--- 2. Running Bus Type Allocation ---")
    allocator = BusTypeAllocator(grid, entropy_model=1)
    bus_types = allocator.allocate(max_iter=50)
    
    # Attach attributes to graph
    nx.set_node_attributes(grid, bus_types, name="bus_type")
    
    print("\n--- 3. Visualizing Bus Types ---")
    viz = GridVisualizer()
    
    print("Opening Bus Type Visualization Plot...")
    # This will show the new plot with:
    # - Red Circles (Gen), Green Triangles (Load), Blue Squares (Conn)
    # - Dashed/Solid/Dotted lines for different connection types
    viz.plot_bus_types(grid, layout='yifan_hu', title="Synthetic Grid: Bus Types & Link Analysis")

if __name__ == "__main__":
    main()