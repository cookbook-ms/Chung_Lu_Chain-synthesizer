====================================
Synthetic Generation and Load Settings 
====================================
In previous pages, we explained how the grid topology can be generated using a Chung-Lu-Chain graph model and the bus types can be assigned using statistics from real-world grids. 

In this page, we explain how the settings of generation and load buses in synthetic grid modeling can be done based on `Elyas et al. (2017) <https://arxiv.org/pdf/1706.09294>`_. The idea is to using the exponential distribution of individual geneartion capacity (or load settings) in a grid, and the non-trivial correlation between the generation capacity (or load setting) and the nodal degree of a generation (or load) bus. The methods are implemented in `capacity_allocator.py <../autoapi/powergrid_synth/capacity_allocator/index.html>`_