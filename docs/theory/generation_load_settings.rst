====================================
Synthetic Generation and Load Settings 
====================================
In previous pages, we explained how the grid topology can be generated using a Chung-Lu-Chain graph model and the bus types can be assigned using statistics from real-world grids. 

In this page, we explain how the settings of generation and load buses in synthetic grid modeling can be done based on `Elyas et al. (2017) <https://arxiv.org/pdf/1706.09294>`_. The idea is to using the exponential distribution of individual generation capacities (or load settings) in a grid, and the non-trivial correlation between the generation capacity (or load setting) and the nodal degree of a generation (or load) bus. The methods are implemented in `capacity_allocator.py <../autoapi/powergrid_synth/capacity_allocator/index.html>`_

Authors in :cite:p:`elyas2017statistical` first analyzed the statistics of the generation capacities and their correlation with topology metrics of the grid, then proposed to determine the generation capcacity via a statistcal approach. A similar approach was doen for the load settings. 

---------------------
Power system modeling
---------------------
Consider an electrical topology of a power grid with $N$ buses and $M$ branches representing transmission lines and transformers. Denote the admittance matrix by $\mathbf{Y}_{N\times N}$, defined as 

.. math:: \mathbf{Y} = \mathbf{A}^\top \Lambda^{-1}(z_l) \mathbf{A}

where $\mathbf{A}$ is the **edge-node incidence matrix**. For a given power network with undirected and unitary links, $\Lambda^{-1}(\cdot)$ is the diagonal inverse matrix with a vector $\cdot$, and $z_l$ the vector of branch impedances in a grid. 

Then the power flow distribuion in a grid follows its network constraints as 

.. math:: \mathbf{P}(t) & = \mathbf{B}^\prime(t) \theta(t) \\ \mathbf{F}(t) & = \Lambda(y_l) \mathbf{A} \theta(t)

where $\mathbf{P}(t) = [\mathbf{P}_g(t), -\mathbf{P}_l(t), \mathbf{P}_l(t)]^\top$ represents the injected real power from generation, load and connection buses; $\theta(t)$ is the vector of phase angles, and $\mathbf{F}(t)$ the real-power delivered along the branches. 

More importantly, grid operations also needs to account for the constraints of generation capcacity, load settings and transmission capcacity such as 

.. math:: P_g^\max \leq & P_g \leq P_g^\max \\ P_l^\max \leq & P_l \leq P_l^\max \\ F^\max \leq & F \leq F_c^\max


--------------------------------
Statistics of Generation setting
--------------------------------
Authors first examined the statistical features of generation capacities in realistic grids in terms of aggregated generation capacity, distribution of individual capacities and their non-trivial correlation with nodal degrees. 

A scaling function of aggreated generation capacity in a grid w.r.t. its network size is derived as 






.. bibliography::
 