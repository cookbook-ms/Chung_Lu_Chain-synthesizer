Synthetic graph model based on [Chung-Lu-Chain model](https://arxiv.org/abs/1711.11098)
================================================================================

![CI Status](https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This generates the graphs based on Chung-Lu-Chain graph model for electric infrastructure networks, given as inputs the degree distribution and diameter.

## Features

* **Multi-Level Generation:** Create grids with arbitrary voltage levels (e.g., High, Medium, Low Voltage).
* **Topological Algorithms:** Implements robust algorithms for:
    * Degree sequence inflation and box assignment (Algorithm 1).
    * Connectivity generation using Diameter Paths and Chung-Lu models (Algorithm 2).
    * Hierarchical interconnections using k-Stars models (Algorithm 3).
* **Advanced Parameter Configuration:**
    * Support for **Poisson**, **Discrete Generalized Log-Normal (DGLN)**, and **Discrete Power Law (DPL)** degree distributions.
    * Automatic parameter optimization to match target average degrees.
* **Visualization:**
    * **Yifan Hu** (Force-directed with adaptive cooling).
    * **Voltage Layered** (Hierarchical view).
    * **Kamada-Kawai** (Topology view).
* **Analysis & Comparison:**
    * Compare generated grids against real-world datasets or reference models (if the real-world graph is provided).
    * Analyze metrics globally or per voltage level.

## Installation

1.  Clone the repository.
2.  Install the package in editable mode:
```bash
    pip install -e .
```
*Requirements:* `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`.

## Quick Start

Here is how to generate a simple 3-level grid, optimize the input distributions, and visualize the result.
```python
    from powergrid_synth.generator import PowerGridGenerator
    from powergrid_synth.input_configurator import InputConfigurator
    from powergrid_synth.visualization import GridVisualizer

    # 1. Configuration
    # Define the voltage levels (Node count, Avg Degree, Diameter, Distribution Type)
    level_specs = [
        {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},    # Backbone (Log-Normal)
        {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},     # Distribution (Power Law)
        {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'}  # Local (Poisson)
    ]

    # Define connections between levels (k-stars model)
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }

    # 2. Generate Parameters
    config = InputConfigurator(seed=42)
    params = config.create_params(level_specs, connection_specs)

    # 3. Generate Topology
    gen = PowerGridGenerator(seed=42)
    grid = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees']
    )

    # 4. Visualize
    viz = GridVisualizer()
    viz.plot_grid(grid, layout='yifan_hu', title="Synthetic Grid")
```
## Module Overview

### Core Generators
* `src/powergrid_synth/generator.py`: The main orchestrator class `PowerGridGenerator`.
* `src/powergrid_synth/preprocessing.py`: **Algorithm 1** - Sets up the "backbone" and spatial boxes for nodes.
* `src/powergrid_synth/edge_creation.py`: **Algorithm 2** - Connects nodes within the same voltage level using diameter paths and probabilistic edges.
* `src/powergrid_synth/transformer_edges.py`: **Algorithm 3** - Connects different voltage levels using a bipartite matching or k-stars approach.

### Configuration & Input
* `src/powergrid_synth/input_configurator.py`: Helper to generate complex input parameters (degrees, transformer logic) from high-level specs.
* `src/powergrid_synth/deg_dist_optimizer.py`: Solves for the optimal alpha, beta (DGLN) or gamma (Power Law) parameters to match a target average degree.

### Analysis & Tools
* `src/powergrid_synth/visualization.py`: Custom plotting routines including the Yifan Hu layout implementation.
* `src/powergrid_synth/analysis.py`: Basic topological metrics (Diameter, Clustering, Path Length).
* `src/powergrid_synth/hierarchical_analysis.py`: Automates analysis per voltage level.
* `src/powergrid_synth/comparison.py`: Tools to compare synthetic grids side-by-side with reference graphs.

## Algorithms Implemented

This package is based on a generative model pipeline:
1.  **Preprocessing:** Inflates degree sequences to account for isolates and assigns nodes to "spatial boxes" to guarantee connectivity diameter.
2.  **CLC (Chung-Lu-Chain):** Generates edges within boxes using a Chung-Lu model, ensuring the specific diameter constraints are met.
3.  **k-Stars Transformer Model:** Models inter-level connections (transformers) as disjoint stars, where the number of stars is proportional to the size of the subgraphs $(h(n_i, n_j) \approx c * \min(n_i, n_j))$.
