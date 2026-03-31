# Getting Started

**Source code:** [GitHub repository](https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer)

## Installation

### From PyPI

```bash
pip install powergridsynth
```

### From source (development mode)

1. Clone the repository:
```bash
git clone https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer.git
cd Chung_Lu_Chain-synthesizer
```

2. Create and activate a virtual environment (e.g. with `uv`):
```bash
uv venv powergrid
source powergrid/bin/activate
```

3. Install in editable mode with development dependencies:
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

See the `examples/` directory for Jupyter notebooks covering each step in detail.

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

## Testing

```bash
pytest tests/ -v
```

## Building Documentation

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` to view the compiled documentation.