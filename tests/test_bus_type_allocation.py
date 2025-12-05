
import sys
import os
import networkx as nx
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.bus_type_allocator import BusTypeAllocator

def main():
    print("--- 1. Generating 3-Level Grid ---")
    configurator = InputConfigurator(seed=42)
    
    level_specs = [
        {'n': 50, 'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},
        {'n': 100, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},
        {'n': 200, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'}
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
    
    print("\n--- 2. Running Bus Type Allocation (AIS Algorithm) ---")
    allocator = BusTypeAllocator(grid, entropy_model=1)
    bus_types = allocator.allocate(max_iter=100)
    
    print("\n--- 3. Results Analysis ---")
    counts = Counter(bus_types.values())
    total = sum(counts.values())
    
    print("Assigned Types:")
    for btype, count in counts.items():
        ratio = count / total
        print(f"  {btype}: {count} ({ratio:.1%})")
        
    # Verify assignment logic
    # Check if Generators are assigned to node 0 (just to see a sample)
    sample_nodes = list(grid.nodes())[:5]
    print("\nSample Assignments:")
    for n in sample_nodes:
        print(f"  Node {n}: {bus_types[n]}")

    # Attach to graph
    nx.set_node_attributes(grid, bus_types, name="bus_type")
    print("\nAttributes attached to NetworkX graph successfully.")

if __name__ == "__main__":
    main()