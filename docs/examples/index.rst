Examples
=============

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Topology Generation <TopologyGeneration>

Here is a complete example showing how to configure, generate, and visualize a synthetic power grid.

.. code-block:: python
   :linenos:

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



