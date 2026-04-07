"""
Complete power-grid synthesis workflow.

This module exposes a single public function :func:`synthesize` that runs
the full CLC pipeline — topology generation, bus-type assignment,
generation/load allocation, dispatch, transmission-line parameterization,
and export — without any intermediate visualisation.

Two operation modes are supported:

* **Mode I – reference-based** (``mode="reference"``):
  Load an existing pandapower network (e.g. IEEE 118-bus), extract its
  topological parameters, and generate a structurally similar synthetic
  clone.

* **Mode II – synthetic** (``mode="synthetic"``):
  Build a grid entirely from user-specified voltage-level specifications
  (node counts, average degrees, diameters, degree distributions) and
  inter-level connection parameters.

Example — Mode I (reference-based)::

    from powergrid_synth.synthesize import synthesize

    synthesize(
        mode="reference",
        reference_case="case118",
        seed=42,
        output_dir="output",
        export_formats=["json", "cgmes"],
    )

Example — Mode II (fully synthetic)::

    from powergrid_synth.synthesize import synthesize

    synthesize(
        mode="synthetic",
        level_specs=[
            {"n": 50,  "avg_k": 3.5, "diam": 10, "dist_type": "dgln"},
            {"n": 150, "avg_k": 2.5, "diam": 15, "dist_type": "dpl"},
            {"n": 300, "avg_k": 2.0, "diam": 20, "dist_type": "poisson"},
        ],
        connection_specs={
            (0, 1): {"type": "k-stars", "c": 0.174, "gamma": 4.15},
            (1, 2): {"type": "k-stars", "c": 0.15,  "gamma": 4.15},
        },
        seed=42,
        output_dir="output",
        export_formats=["json", "matpower"],
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

from .bus_type_allocator import BusTypeAllocator
from .capacity_allocator import CapacityAllocator
from .generation_dispatcher import GenerationDispatcher
from .generator import PowerGridGenerator
from .input_configurator import InputConfigurator
from .input_extractor import extract_topology_params_from_graph
from .load_allocator import LoadAllocator
from .transmission import TransmissionLineAllocator


def _get_pandapower_cases() -> Dict[str, object]:
    try:
        import pandapower.networks as pn
    except ImportError as exc:
        raise ImportError(
            "synthesize() in reference mode requires pandapower. "
            "Install it with: pip install powergrid_synth[export]"
        ) from exc

    return {name: getattr(pn, name) for name in dir(pn) if name.startswith("case")}

# Maps user-facing format names to (GridExporter method, default extension).
_EXPORT_DISPATCH = {
    "json":     ("to_json",      ".json"),
    "excel":    ("to_excel",     ".xlsx"),
    "sqlite":   ("to_sqlite",    ".sqlite"),
    "pickle":   ("to_pickle",    ".p"),
    "xiidm":    ("to_pypowsybl", ".xiidm"),
    "cgmes":    ("to_cgmes",     "_cgmes"),
    "matpower": ("to_matpower",  ""),
    "psse":     ("to_psse",      ""),
}


def synthesize(
    *,
    mode: str,
    # --- Mode I (reference) ---
    reference_case: Optional[str] = None,
    reference_net=None,
    # --- Mode II (synthetic) ---
    level_specs: Optional[List[Dict]] = None,
    connection_specs: Optional[Dict[Tuple[int, int], Dict]] = None,
    # --- Common parameters ---
    seed: Optional[int] = None,
    keep_lcc: bool = True,
    entropy_model: int = 0,
    bus_type_ratio: Optional[List[float]] = None,
    ref_sys_id: int = 1,
    loading_level: str = "H",
    refine_topology: bool = False,
    base_kv_map: Optional[Dict[int, float]] = None,
    output_dir: str = "output",
    output_name: str = "synthetic_grid",
    export_formats: Sequence[str] = ("json",),
) -> nx.Graph:
    """Run the full CLC synthesis pipeline and export the result.

    Parameters
    ----------
    mode : ``"reference"`` or ``"synthetic"``
        Selects the operation mode.

        * ``"reference"`` — extract topology parameters from an existing
          pandapower network (Mode I).
        * ``"synthetic"`` — generate topology from user-provided level /
          connection specs (Mode II).

    reference_case : str, optional
        Name of a pandapower built-in network (e.g. ``"case118"``).
        Used only when ``mode="reference"``.  Ignored if *reference_net*
        is given.
    reference_net : pandapowerNet, optional
        A pre-loaded pandapower network.  Takes precedence over
        *reference_case* when ``mode="reference"``.

    level_specs : list of dict, optional
        Voltage-level specifications for Mode II.  Each dict must have
        keys ``"n"`` (int), ``"avg_k"`` (float), ``"diam"`` (int), and
        ``"dist_type"`` (``"dgln"`` | ``"dpl"`` | ``"poisson"``).
        Optionally ``"max_k"`` (int).
    connection_specs : dict, optional
        Transformer connection specs for Mode II.  Maps
        ``(level_i, level_j)`` tuples to dicts with ``"type"``
        (``"k-stars"`` | ``"simple"``) and parameters ``"c"``, ``"gamma"``.

    seed : int, optional
        Random seed for reproducibility.
    keep_lcc : bool
        If ``True`` (default), keep only the largest connected component
        after topology generation.
    entropy_model : int
        Bus-type entropy model — 0 (standard) or 1 (weighted).
    bus_type_ratio : list of float, optional
        Target ``[Gen%, Load%, Conn%]`` ratios.  If ``None``, default
        ratios based on network size are used.
    ref_sys_id : int
        Reference system for statistical tables (0–3).
    loading_level : str
        Load level — ``"D"`` (deterministic), ``"L"`` (light),
        ``"M"`` (medium), ``"H"`` (heavy, default).
    refine_topology : bool
        If ``True``, the transmission allocator may add/remove edges to
        improve DCPF convergence.
    base_kv_map : dict, optional
        Custom ``{level_index: kV}`` mapping.  If ``None`` and
        ``mode="reference"``, the mapping is extracted from the reference
        grid.
    output_dir : str
        Directory for exported files (created if needed).
    output_name : str
        Base filename (without extension) for exported files.
    export_formats : sequence of str
        One or more format names to export.  Supported:
        ``"json"``, ``"excel"``, ``"sqlite"``, ``"pickle"``,
        ``"xiidm"``, ``"cgmes"``, ``"matpower"``, ``"psse"``.

    Returns
    -------
    nx.Graph
        The fully parameterised synthetic grid (a
        :class:`~powergrid_synth.grid_graph.PowerGridGraph`).

    Raises
    ------
    ValueError
        If *mode* is not ``"reference"`` or ``"synthetic"``, or if
        required arguments for the chosen mode are missing.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    mode = mode.lower().strip()
    if mode not in ("reference", "synthetic"):
        raise ValueError(
            f"mode must be 'reference' or 'synthetic', got {mode!r}"
        )

    if mode == "reference":
        if reference_net is None and reference_case is None:
            raise ValueError(
                "Mode 'reference' requires either reference_case or "
                "reference_net."
            )
    else:
        if level_specs is None or connection_specs is None:
            raise ValueError(
                "Mode 'synthetic' requires both level_specs and "
                "connection_specs."
            )

    # ------------------------------------------------------------------
    # 1. Obtain topology parameters
    # ------------------------------------------------------------------
    if mode == "reference":
        from .data_format_converter import pandapower_to_nx

        if reference_net is not None:
            net = reference_net
        else:
            pandapower_cases = _get_pandapower_cases()
            factory = pandapower_cases.get(reference_case)
            if factory is None:
                raise ValueError(
                    f"Unknown pandapower case {reference_case!r}. "
                    f"Available: {sorted(pandapower_cases)}"
                )
            net = factory()

        graph_ref = pandapower_to_nx(net)
        params = extract_topology_params_from_graph(graph_ref)

        # Use the reference grid's base_kv_map unless the user overrides.
        if base_kv_map is None:
            base_kv_map = graph_ref.graph.get("base_kv_map")

        print(
            f"[1] Loaded reference grid: "
            f"{graph_ref.number_of_nodes()} nodes, "
            f"{graph_ref.number_of_edges()} edges."
        )

    else:  # mode == "synthetic"
        configurator = InputConfigurator(seed=seed)
        params = configurator.create_params(level_specs, connection_specs)
        print(f"[1] Generated synthetic input parameters for {len(level_specs)} voltage levels.")

    # ------------------------------------------------------------------
    # 2. Generate topology
    # ------------------------------------------------------------------
    gen = PowerGridGenerator(seed=seed)
    graph = gen.generate_grid(
        degrees_by_level=params["degrees_by_level"],
        diameters_by_level=params["diameters_by_level"],
        transformer_degrees=params["transformer_degrees"],
        keep_lcc=keep_lcc,
    )
    print(
        f"[2] Topology generated: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges."
    )

    # ------------------------------------------------------------------
    # 3. Assign bus types
    # ------------------------------------------------------------------
    allocator = BusTypeAllocator(
        graph,
        entropy_model=entropy_model,
        bus_type_ratio=bus_type_ratio,
    )
    bus_types = allocator.allocate()
    nx.set_node_attributes(graph, bus_types, name="bus_type")
    _print_bus_types(bus_types)

    # ------------------------------------------------------------------
    # 4. Allocate generation capacities
    # ------------------------------------------------------------------
    cap_alloc = CapacityAllocator(graph, ref_sys_id=ref_sys_id)
    capacities = cap_alloc.allocate()
    nx.set_node_attributes(graph, capacities, name="pg_max")
    total_cap = sum(capacities.values())
    print(f"[4] Generation capacities: total = {total_cap:.1f} MW")

    # ------------------------------------------------------------------
    # 5. Allocate loads
    # ------------------------------------------------------------------
    load_alloc = LoadAllocator(graph, ref_sys_id=ref_sys_id)
    loads = load_alloc.allocate(loading_level=loading_level)
    nx.set_node_attributes(graph, loads, name="pl")
    total_load = sum(loads.values())
    print(
        f"[5] Loads allocated: total = {total_load:.1f} MW "
        f"({total_load / total_cap:.0%} loading)"
    )

    # ------------------------------------------------------------------
    # 6. Dispatch generation
    # ------------------------------------------------------------------
    dispatcher = GenerationDispatcher(graph, ref_sys_id=ref_sys_id)
    dispatch = dispatcher.dispatch()
    nx.set_node_attributes(graph, dispatch, name="pg")
    total_gen = sum(dispatch.values())
    print(
        f"[6] Generation dispatched: {total_gen:.1f} MW "
        f"(reserve {total_cap - total_gen:.1f} MW)"
    )

    # ------------------------------------------------------------------
    # 7. Allocate transmission lines (impedance + capacity)
    # ------------------------------------------------------------------
    trans_alloc = TransmissionLineAllocator(graph, ref_sys_id=ref_sys_id)
    line_caps = trans_alloc.allocate(refine_topology=refine_topology)
    n_lines = len(line_caps)
    avg_cap = sum(line_caps.values()) / n_lines if n_lines else 0
    print(
        f"[7] Transmission lines: {n_lines} lines, "
        f"avg capacity = {avg_cap:.1f} MVA"
    )

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    from .exporter import GridExporter

    os.makedirs(output_dir, exist_ok=True)
    exporter = GridExporter(graph, base_kv_map=base_kv_map)

    for fmt in export_formats:
        fmt_lower = fmt.lower().strip()
        if fmt_lower not in _EXPORT_DISPATCH:
            print(f"  [!] Unknown format {fmt!r}, skipping. "
                  f"Available: {sorted(_EXPORT_DISPATCH)}")
            continue

        method_name, default_ext = _EXPORT_DISPATCH[fmt_lower]
        filepath = os.path.join(output_dir, output_name + default_ext)
        method = getattr(exporter, method_name)

        if fmt_lower == "xiidm":
            method(filepath, format="XIIDM")
        else:
            method(filepath)

        print(f"  -> Exported {fmt_lower}: {filepath}")

    print("[8] Done.")
    return graph


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _print_bus_types(bus_types: Dict[int, str]) -> None:
    from collections import Counter
    counts = Counter(bus_types.values())
    total = sum(counts.values())
    parts = ", ".join(
        f"{t}: {counts[t]} ({counts[t] / total:.0%})"
        for t in ("Gen", "Load", "Conn")
        if t in counts
    )
    print(f"[3] Bus types assigned: {parts}")
