r"""
This module aims to convert the generated data, including the grid topology (nx.Graph), and other data like bus types, generations and loads, into the other required data formats: 

- powersybl
- pandapower
- ... 

or the other way around
"""

import networkx as nx
import pandas as pd
import numpy as np

def pandapower_to_nx(net) -> nx.Graph:
    """
    TODO: pandapower.topology.create_graph might be able to do a better job? 

    Converts a Pandapower network to a NetworkX graph compatible with powergrid_synth.
    
    It maps:
    1. Voltage levels (float kV) -> Discrete levels (0, 1, 2... based on hierarchy).
    2. Components (Gen/Sgen/Ext_grid/Load) -> Bus Types ('Gen', 'Load', 'Conn').
    
    Args:
        net: The pandapower network object.
        
    Returns:
        nx.Graph: The graph with 'voltage_level' and 'bus_type' node attributes.
    """
    G = nx.Graph()
    
    # 1. Voltage Mapping (Highest kV = Level 0)
    # Filter out NaNs if any
    valid_voltages = [v for v in net.bus.vn_kv.unique() if not pd.isna(v)]
    unique_voltages = sorted(valid_voltages, reverse=True)
    vol_to_level = {v: i for i, v in enumerate(unique_voltages)}
    
    # 2. Identify Bus Types
    # Generators: connected to gen, sgen, or ext_grid
    gen_buses = set(net.gen.bus.values) | set(net.ext_grid.bus.values) | set(net.sgen.bus.values)
    
    # Loads: connected to load (exclude buses that are already Gens)
    load_buses = set(net.load.bus.values) - gen_buses
    
    # 3. Add Nodes
    for idx, row in net.bus.iterrows():
        # Handle voltage level
        vn_kv = row['vn_kv']
        lvl = vol_to_level.get(vn_kv, 0) # Default to 0 if unknown
        
        # Determine Type
        if idx in gen_buses:
            b_type = 'Gen'
        elif idx in load_buses:
            b_type = 'Load'
        else:
            b_type = 'Conn'
            
        G.add_node(idx, 
                   voltage_level=lvl, 
                   vn_kv=vn_kv, 
                   bus_type=b_type,
                   name=row.get('name', f'Bus {idx}'))

    # 4. Add Edges (Lines)
    for _, row in net.line.iterrows():
        if row.get('in_service', True):
            G.add_edge(row['from_bus'], row['to_bus'], type='line')
            
    # 5. Add Edges (Transformers)
    for _, row in net.trafo.iterrows():
        if row.get('in_service', True):
            G.add_edge(row['hv_bus'], row['lv_bus'], type='transformer')
            
    return G