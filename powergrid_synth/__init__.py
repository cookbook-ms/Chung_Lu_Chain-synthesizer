"""
PowerGridSynth Package
======================

The main package for PowerGridSynth. 

This package contains submodules for generating, analyzing, and exporting 
synthetic power grids.
"""
from .generator import PowerGridGenerator
from .input_configurator import InputConfigurator
from .bus_type_allocator import BusTypeAllocator
from .capacity_allocator import CapacityAllocator
from .load_allocator import LoadAllocator
from .generation_dispatcher import GenerationDispatcher
from .transmission import TransmissionLineAllocator
from .visualization import GridVisualizer
from .reference_data import get_reference_stats
from .comparison import GraphComparator
from .grid_graph import PowerGridGraph
from .hierarchical_analysis import HierarchicalAnalyzer
from .input_extractor import extract_topology_params_from_graph


# ---------------------------------------------------------------------------
# Lazy imports for modules that require optional dependencies
# (pandapower, pypowsybl, lightsim2grid, numba, pypowsybl-jupyter).
# The real import is deferred until the name is actually used, so that
# users who only need the core synthesis pipeline are never blocked by
# missing optional packages.
# ---------------------------------------------------------------------------

def __getattr__(name):
    _LAZY = {
        "synthesize": (".synthesize", "synthesize"),
        "GridExporter": (".exporter", "GridExporter"),
        "pandapower_to_nx": (".data_format_converter", "pandapower_to_nx"),
        "nx_to_pandapower": (".data_format_converter", "nx_to_pandapower"),
        "pandapower_to_pypowsybl": (
            ".data_format_converter",
            "pandapower_to_pypowsybl",
        ),
    }

    if name in _LAZY:
        module_path, attr = _LAZY[name]
        _EXTRA_DEPS = {
            "synthesize": "pandapower",
            "GridExporter": "pandapower",
            "pandapower_to_nx": "pandapower",
            "nx_to_pandapower": "pandapower",
            "pandapower_to_pypowsybl": "pypowsybl",
        }
        try:
            import importlib
            mod = importlib.import_module(module_path, __name__)
            obj = getattr(mod, attr)
            # Cache in the module namespace so __getattr__ is only called once
            globals()[name] = obj
            return obj
        except ImportError as exc:
            pkg = _EXTRA_DEPS.get(name, "the required optional package")
            raise ImportError(
                f"{name!r} requires {pkg}. "
                f"Install it with:  pip install powergrid_synth[export]"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (always available)
    "PowerGridGenerator",
    "InputConfigurator",
    "BusTypeAllocator",
    "CapacityAllocator",
    "LoadAllocator",
    "GenerationDispatcher",
    "TransmissionLineAllocator",
    "GridVisualizer",
    "get_reference_stats",
    "GraphComparator",
    "PowerGridGraph",
    "HierarchicalAnalyzer",
    "extract_topology_params_from_graph",
    # Optional (lazy-loaded, require pandapower / pypowsybl)
    "synthesize",
    "GridExporter",
    "pandapower_to_nx",
    "nx_to_pandapower",
    "pandapower_to_pypowsybl",
]

__version__ = "0.1.0"