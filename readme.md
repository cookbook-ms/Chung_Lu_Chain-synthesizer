# PowerGridSynth

![CI Status](https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python package for generating realistic synthetic power grids with statistically accurate topology, bus types, generation/load settings, and transmission-line parameters.

The pipeline starts from a [Chung-Lu-Chain (CLC)](https://arxiv.org/abs/1711.11098) graph model that reproduces prescribeddegree distributions and diameters across multiple voltage levels, then layers on bus-type assignment, generation/load capacity allocation, generation dispatch, and transmission-line impedance/capacity assignment drawn from empirical statistics of real grids (NYISO, WECC).

## Documentation

Full documentation (theory, API reference, tutorials): [power-grid-synthesizer.readthedocs.io](https://power-grid-synthesizer.readthedocs.io/en/latest/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer.git
cd Chung_Lu_Chain-synthesizer
```
2. Create and activate a virtual environment (e.g. with `uv`):
```bash
uv venv powergrid_studio
source powergrid_studio/bin/activate
```
3. Install in editable mode:
```bash
uv pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`, `pandapower`, `lightsim2grid`, `pypowsybl`.

## Quick Start

```python
from powergrid_synth import (
    InputConfigurator, PowerGridGenerator, BusTypeAllocator,
    CapacityAllocator, LoadAllocator, GenerationDispatcher,
    TransmissionLineAllocator, GridVisualizer,
)

# 1. Configure voltage levels and inter-level connections
level_specs = [
    {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},
    {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},
    {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'},
]
connection_specs = {
    (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
    (1, 2): {'type': 'k-stars', 'c': 0.15,  'gamma': 4.15},
}

config = InputConfigurator(seed=42)
params = config.create_params(level_specs, connection_specs)

# 2. Generate topology (CLC model)
gen = PowerGridGenerator(seed=42)
grid = gen.generate_grid(
    degrees_by_level=params['degrees_by_level'],
    diameters_by_level=params['diameters_by_level'],
    transformer_degrees=params['transformer_degrees'],
)

# 3. Assign bus types, generation/load, dispatch, and line parameters
BusTypeAllocator(grid).allocate()
CapacityAllocator(grid).allocate()
LoadAllocator(grid).allocate()
GenerationDispatcher(grid).dispatch()
TransmissionLineAllocator(grid).allocate()

# 4. Visualize
GridVisualizer().plot_grid(grid, layout='yifan_hu', title="Synthetic Grid")
```

See `examples/` for Jupyter notebooks covering each step in detail.

## Synthesis Pipeline

| Step | Module | Method |
|------|--------|--------|
| **1. Topology generation** | `generator.py`, `preprocessing.py`, `edge_creation.py`, `transformer_edges.py` | CLC graph model with prescribed degree distribution and diameter per voltage level; k-stars inter-level transformer model |
| **2. Bus-type assignment** | `bus_type_allocator.py` | Entropy-based Artificial Immune System (AIS) optimisation reproducing empirical 2-D joint distribution of bus types and node degrees |
| **3. Generation capacity** | `capacity_allocator.py` | Exponential/extreme-value sampling with capacity–degree correlation via 2-D PMF table |
| **4. Load allocation** | `load_allocator.py` | Empirical 2-D probability table matching load–degree joint distribution |
| **5. Generation dispatch** | `generation_dispatcher.py` | Three-category partitioning (uncommitted / partially committed / fully committed) with 2-D bin matching and iterative balancing |
| **6. Transmission lines** | `transmission.py`, `dcpf.py` | LogNormal impedance magnitudes, Lévy-stable line angles, DCPF-based swapping, exponential gauge-ratio assignment via 2-D table, optional topology refinement |

Statistical parameters are stored in `reference_data.py` for reference systems (NYISO-2935, WECC-16994).

## Module Overview

### Topology Generation
| Module | Description |
|--------|-------------|
| `generator.py` | Main orchestrator — `PowerGridGenerator` |
| `preprocessing.py` | Backbone setup and spatial-box assignment (Algorithm 1) |
| `edge_creation.py` | Intra-level edge creation via CLC model (Algorithm 2) |
| `transformer_edges.py` | Inter-level connections via k-stars model (Algorithm 3) |
| `input_configurator.py` | High-level specification → detailed input parameters |
| `deg_dist_optimizer.py` | Optimises distribution parameters (DGLN α,β / Power-Law γ) to match target average degree |

### Electrical Assignment
| Module | Description |
|--------|-------------|
| `bus_type_allocator.py` | AIS-based Generator / Load / Connection bus-type assignment |
| `capacity_allocator.py` | Generation capacity (Pg_max) allocation |
| `load_allocator.py` | Active-power load (Pl) allocation |
| `generation_dispatcher.py` | Dispatch factor (α) assignment and generation–load balancing |
| `transmission.py` | Branch impedance (R, X) and thermal capacity (F_max) allocation |
| `dcpf.py` | Lightweight DC power-flow solver |
| `reference_data.py` | Pre-computed empirical statistics for NYISO, WECC reference systems |

### Analysis, Export & Visualisation
| Module | Description |
|--------|-------------|
| `grid_graph.py` | `PowerGridGraph` — NetworkX Graph subclass with voltage-level filtering |
| `analysis.py` | Topological metrics (diameter, clustering, path length) |
| `hierarchical_analysis.py` | Per-voltage-level analysis |
| `comparison.py` | Side-by-side comparison of synthetic vs. reference grids |
| `visualization.py` | Plotting routines (Yifan Hu layout, per-level views, interactive dashboards) |
| `exporter.py` | Export to JSON, Excel, MATPOWER, CGMES, XIIDM, PSS/E |
| `data_format_converter.py` | Conversion between NetworkX, pandapower, pypowsybl |

## Examples

| Notebook | Topic |
|----------|-------|
| `examples/TopologyGeneration.ipynb` | CLC topology generation and input configuration |
| `examples/BusTypeAssignment.ipynb` | AIS bus-type allocation |
| `examples/GenLoadSettings.ipynb` | Generation capacity and load allocation |
| `examples/ieee_test.ipynb` | Validation against IEEE test cases |
| `examples/grid2graph.ipynb` | Converting real grid data to graph representation |
| `examples/hodge_analysis.ipynb` | Hodge-theoretic flow analysis |

## Testing

```bash
pytest tests/ -v
```

## References

1. Aksoy et al. (2019). "A Generative Graph Model for Electrical Infrastructure Networks." *Journal of Complex Networks*, 7(1), 128–162. [arXiv:1711.11098](https://arxiv.org/abs/1711.11098)
2. Elyas & Wang (2017). "Improved Synthetic Power Grid Modeling With Correlated Bus Type Assignments." *IEEE Trans. Power Syst.*, 32(5), 3391–3402. [DOI:10.1109/TPWRS.2016.2636165](https://ieeexplore.ieee.org/document/7763878)
3. Elyas et al. (2017). "On the Statistical Settings of Generation and Load in Synthetic Grid Modeling." [arXiv:1706.09294](https://arxiv.org/abs/1706.09294)
4. Sadeghian et al. (2018). "A Novel Algorithm for Statistical Assignment of Transmission Capacities in Synthetic Grid Modeling." *IEEE PESGM*. [DOI:10.1109/PESGM.2018.8585532](https://ieeexplore.ieee.org/document/8585532)

## License

MIT
