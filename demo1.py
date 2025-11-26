import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the 'src' directory to the path so we can import the package locally
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.generator import PowerGridGenerator
from powergrid_synth.input_configurator import InputConfigurator
from powergrid_synth.hierarchical_analysis import HierarchicalAnalyzer

def main():
    print("--- 1. Configuration: Setting up 5-Level Hierarchy ---")
    
    # Initialize Configurator
    configurator = InputConfigurator(seed=42)

    # Define Specs for 5 Voltage Levels
    # We use a mix of distributions to simulate realistic grid topology
    level_specs = [
        # Level 0: Extra High Voltage (Backbone) - Log-Normal for hubs
        {'n': 50, 'avg_k': 4.0, 'diam': 8, 'dist_type': 'dgln', 'max_k': 20},
        
        # Level 1: High Voltage - Power Law for scale-free structure
        {'n': 100, 'avg_k': 3.0, 'diam': 12, 'dist_type': 'dpl', 'max_k': 20},
        
        # Level 2: Medium Voltage - Poisson
        {'n': 200, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'poisson'},
        
        # Level 3: Low Voltage - Poisson
        {'n': 400, 'avg_k': 2.2, 'diam': 20, 'dist_type': 'poisson'},
        
        # Level 4: Residential - Poisson (Radial-like)
        {'n': 800, 'avg_k': 1.8, 'diam': 25, 'dist_type': 'poisson'}
    ]

    # Define k-stars connections between levels
    # (source_level, target_level): {type, c, gamma}
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (2, 3): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15},
        (3, 4): {'type': 'k-stars', 'c': 0.10, 'gamma': 4.15}
    }

    print("--- 2. Generating Input Parameters ---")
    params = configurator.create_params(level_specs, connection_specs)

    print("--- 3. Generating Grid Topology ---")
    gen = PowerGridGenerator(seed=42)
    grid_graph = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees']
    )
    
    print(f"Grid Generated: {grid_graph.number_of_nodes()} nodes, {grid_graph.number_of_edges()} edges")

    print("\n--- 4. Running Hierarchical Analysis ---")
    print("This will print topological metrics and plot degree distributions.")
    
    # Initialize the Analyzer
    analyzer = HierarchicalAnalyzer(grid_graph)
    
    # Run the full report
    # - Calculates global metrics (Diameter, Clustering, etc.)
    # - Calculates metrics for EACH voltage level subgraph
    # - Plots degree distributions (Log-Log scale)
    analyzer.run_full_report(log_scale=True)
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()