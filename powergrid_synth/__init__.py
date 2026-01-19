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
from .exporter import GridExporter
from .reference_data import get_reference_stats
from .comparison import GraphComparator
from .grid_graph import PowerGridGraph


__all__ = [
    "PowerGridGenerator",
    "InputConfigurator",
    "BusTypeAllocator",
    "CapacityAllocator",
    "LoadAllocator",
    "GenerationDispatcher",
    "TransmissionLineAllocator",
    "GridVisualizer",
    "GridExporter",
    "get_reference_stats",
    "GraphComparator",
    "PowerGridGraph"
]

__version__ = "0.1.0"