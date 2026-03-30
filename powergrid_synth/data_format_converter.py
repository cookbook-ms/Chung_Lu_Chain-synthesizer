import networkx as nx
import pandas as pd
import numpy as np
import pandapower as pp
import pypowsybl as ppl 


def pandapower_to_nx(net: pp.pandapowerNet) -> nx.Graph:
    """
    Converts a pandapowerNet object into a NetworkX graph compatible with the synthesizer.
    Extracts buses, lines, transformers, loads, and generators.
    """
    G = nx.Graph()
    
    # 1. Voltage Mapping (Highest kV = Level 0)
    # Filter out NaNs if any
    valid_voltages = [v for v in net.bus.vn_kv.unique() if not pd.isna(v)]
    unique_voltages = sorted(valid_voltages, reverse=True)
    vol_to_level = {v: i for i, v in enumerate(unique_voltages)}
    
    # Store the base kv mapping in the graph attributes
    G.graph['base_kv_map'] = {i: v for i, v in enumerate(unique_voltages)}
    
    # 2. Add Buses
    for idx, row in net.bus.iterrows():
        # Default all buses to 'Conn', later overwritten if they have Load/Gen
        lvl = vol_to_level.get(row['vn_kv'], 0)
        G.add_node(idx, voltage_level=lvl, vn_kv=row['vn_kv'], bus_type='Conn',
                   max_vm_pu=row.get('max_vm_pu', 1.05), min_vm_pu=row.get('min_vm_pu', 0.95))
        
    # 3. Assign Load attributes
    if 'load' in net and not net.load.empty:
        for idx, row in net.load.iterrows():
            bus_idx = row['bus']
            G.nodes[bus_idx]['bus_type'] = 'Load'
            G.nodes[bus_idx]['pl'] = row.get('p_mw', 0.0)
            G.nodes[bus_idx]['ql'] = row.get('q_mvar', 0.0)
            
    # 4. Assign Generator attributes (Ext Grids, Static Gens & Standard Gens)
    if 'ext_grid' in net and not net.ext_grid.empty:
        for idx, row in net.ext_grid.iterrows():
            bus_idx = row['bus']
            G.nodes[bus_idx]['bus_type'] = 'Gen'
            # Assign an arbitrarily large capacity for the slack bus
            G.nodes[bus_idx]['pg_max'] = 9999.0
            G.nodes[bus_idx]['is_ext_grid'] = True

    if 'sgen' in net and not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            bus_idx = row['bus']
            G.nodes[bus_idx]['bus_type'] = 'Gen'
            G.nodes[bus_idx]['pg'] = row.get('p_mw', 0.0)
            G.nodes[bus_idx]['qg'] = row.get('q_mvar', 0.0)
            G.nodes[bus_idx]['pg_max'] = row.get('sn_mva', 0.0)
            
    if 'gen' in net and not net.gen.empty:
        for idx, row in net.gen.iterrows():
            bus_idx = row['bus']
            G.nodes[bus_idx]['bus_type'] = 'Gen'
            G.nodes[bus_idx]['pg'] = row.get('p_mw', 0.0)
            G.nodes[bus_idx]['pg_max'] = row.get('sn_mva', 0.0)
            
    # 5. Add Lines
    if 'line' in net and not net.line.empty:
        for idx, row in net.line.iterrows():
            G.add_edge(row['from_bus'], row['to_bus'], 
                       type='line',
                       r=row.get('r_ohm_per_km', 0.0) * row.get('length_km', 1.0),
                       x=row.get('x_ohm_per_km', 0.001) * row.get('length_km', 1.0),
                       c=row.get('c_nf_per_km', 0.0) * row.get('length_km', 1.0),
                       g=row.get('g_us_per_km', 0.0) * row.get('length_km', 1.0),
                       capacity=row.get('max_i_ka', 0.0) * np.sqrt(3) * net.bus.at[row['from_bus'], 'vn_kv'],
                       parallel=row.get('parallel', 1))
                       
    # 6. Add Transformers
    if 'trafo' in net and not net.trafo.empty:
        for idx, row in net.trafo.iterrows():
            G.add_edge(row['hv_bus'], row['lv_bus'], 
                       type='transformer',
                       capacity=row.get('sn_mva', 0.0),
                       parallel=row.get('parallel', 1))
            
    return G

def nx_to_pandapower(graph: nx.Graph, base_mva: float = 100.0, base_kv_map: dict = None) -> pp.pandapowerNet:
    """
    Converts a synthetic NetworkX graph into a pandapowerNet object.
    Uses native Pandapower creation functions to ensure memory safety for the solver.
    
    Args:
        graph: The synthetic power grid (NetworkX Graph) containing electrical properties.
        base_mva: System base MVA for per-unit calculations.
        base_kv_map: Dictionary mapping voltage level indices (0, 1, 2) to actual kV (e.g., {0: 380.0, 1: 110.0}).
    """
    # Create empty network structure
    net = pp.create_empty_network()
    net.sn_mva = base_mva
    
    # Default hierarchy if not specified: HV -> MV -> LV -> Residential
    if base_kv_map is None:
        base_kv_map = {0: 380.0, 1: 110.0, 2: 20.0, 3: 0.4, 4: 0.12}
        
    sorted_nodes = sorted(list(graph.nodes()))
    
    # Identify the slack bus (e.g., the largest generator) for power flow solution
    slack_bus = None
    max_pg = -1.0
    for n in sorted_nodes:
        if graph.nodes[n].get('bus_type') == 'Gen':
            p_cap = graph.nodes[n].get('pg_max', 0.0)
            if p_cap > max_pg:
                max_pg = p_cap
                slack_bus = n
    if slack_bus is None and sorted_nodes:
        slack_bus = sorted_nodes[0]
    
    # --- 1. Map Nodes (Buses, Loads, Gens, Ext Grid) ---
    for n in sorted_nodes:
        d = graph.nodes[n]
        lvl = d.get('voltage_level', 0)
        vn_kv = base_kv_map.get(lvl, 110.0)
        
        # 'b' = busbar, 'n' = node
        b_type = 'n' if d.get('bus_type') == 'Conn' else 'b'
        
        # Create Bus natively, preserving the exact NetworkX node ID as the dataframe index
        pp.create_bus(net, vn_kv=vn_kv, name=f"Bus_{n}", type=b_type, zone=1,
                      max_vm_pu=d.get('max_vm_pu', 1.05), min_vm_pu=d.get('min_vm_pu', 0.95), index=n)
        
        # Parse Loads
        if d.get('bus_type') == 'Load' and d.get('pl', 0) > 0:
            pp.create_load(net, bus=n, p_mw=d.get('pl', 0.0), q_mvar=d.get('ql', 0.0),
                           sn_mva=base_mva, name=f"Load_{n}", type='wye')
            
        # Parse Generators
        if d.get('bus_type') == 'Gen':
            gen_sn_mva = d.get('pg_max', base_mva)
            # Generate random reactive power limits as fractions of rated capacity
            min_q_fraction = np.random.uniform(-0.3, 0.0)  # Can absorb reactive power
            max_q_fraction = np.random.uniform(0.3, 0.7)   # Can generate reactive power
            min_q_mvar = float(min_q_fraction * gen_sn_mva)
            max_q_mvar = float(max_q_fraction * gen_sn_mva)

            if n == slack_bus:
                # Add as External Grid to provide a reference slack bus for power flow
                pp.create_ext_grid(net, bus=n, vm_pu=1.0, va_degree=0.0, name=f"Ext_Grid_{n}",
                                   min_q_mvar=min_q_mvar, max_q_mvar=max_q_mvar)
            else:
                pp.create_gen(net, bus=n, p_mw=d.get('pg', 0.0), vm_pu=d.get('vm_pu', 1.0),
                              sn_mva=gen_sn_mva, name=f"Gen_{n}",
                              min_q_mvar=min_q_mvar, max_q_mvar=max_q_mvar)

    # --- 2. Map Edges (Lines, Transformers) ---
    for u, v, d in graph.edges(data=True):
        edge_type = d.get('type', 'line')
        
        u_lvl = graph.nodes[u].get('voltage_level', 0)
        v_lvl = graph.nodes[v].get('voltage_level', 0)
        
        # It's a transformer if explicitly tagged or if it crosses voltage boundaries
        if edge_type == 'transformer' or u_lvl != v_lvl:
            vn_u = base_kv_map.get(u_lvl, 110.0)
            vn_v = base_kv_map.get(v_lvl, 110.0)
            
            # Orient High Voltage (HV) and Low Voltage (LV) sides
            if vn_u >= vn_v:
                hv_bus, lv_bus = u, v
                vn_hv, vn_lv = vn_u, vn_v
            else:
                hv_bus, lv_bus = v, u
                vn_hv, vn_lv = vn_v, vn_u
                
            sn_mva = d.get('capacity', base_mva)
            
            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus, sn_mva=sn_mva,
                vn_hv_kv=vn_hv, vn_lv_kv=vn_lv, vk_percent=10.0, vkr_percent=0.1,
                pfe_kw=0.0, i0_percent=0.0, name=f"Trafo_{u}_{v}",
                parallel=d.get('parallel', 1)
            )
            
        else: # It's a standard transmission/distribution line
            vn_kv = base_kv_map.get(u_lvl, 110.0)
            
            # Impedance Conversion: PU -> Ohms
            z_base = (vn_kv**2) / base_mva
            r_ohm = d.get('r', 0.0) * z_base
            x_ohm = d.get('x', 0.001) * z_base
            
            # Capacity Conversion: MVA -> kA
            cap_mva = d.get('capacity', 999.0)
            max_i_ka = cap_mva / (np.sqrt(3) * vn_kv)
            
            pp.create_line_from_parameters(
                net, from_bus=u, to_bus=v, length_km=1.0,
                r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=d.get('c', 0.0),
                g_us_per_km=d.get('g', 0.0), max_i_ka=max_i_ka,
                name=f"Line_{u}_{v}", type='ol', parallel=d.get('parallel', 1)
            )

    return net


def pandapower_to_pypowsybl(net: pp.pandapowerNet) -> ppl.network.impl.network.Network:
    """
    Converts a pandapowerNet object into a Pypowsybl Network object.
    """
    ppl_net = ppl.network.impl.pandapower_converter.convert_from_pandapower(net)
    return ppl_net