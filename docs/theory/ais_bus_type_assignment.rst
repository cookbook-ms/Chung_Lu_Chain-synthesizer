Artificial Immune System (AIS) for Bus Type Allocation
======================================================

This document outlines the pseudo-algorithm for the **Artificial Immune System (AIS)** used in ``BusTypeAllocator`` to assign bus types (Generator, Load, Connection) to a grid topology. The goal is to minimize the difference between the generated grid's entropy (:math:`W`) and a target entropy (:math:`W^*`) derived from reference statistics.

Initialization
--------------

**Input:**

* Graph Topology :math:`G(V, E)` with :math:`N` nodes and :math:`M` edges.
* Entropy Model Selection (0 or 1).

**Setup:**

1. **Determine Ratios:** Calculate target percentages for Generators (:math:`P_G`), Loads (:math:`P_L`), and Connections (:math:`P_C`) based on network size :math:`N`.

   * *e.g., If* :math:`N < 2000`: :math:`P_G=23\%, P_L=55\%, P_C=22\%`

2. **Constraint Identification:** Identify non-leaf nodes (degree > 1) as preferred candidates for Connection buses.

Target Entropy Estimation (:math:`W^*`)
----------------------------------------

Before optimization, we estimate the "natural" entropy of the specific topology via Monte Carlo simulation.

**Procedure:**

1. Initialize empty list ``samples``.

2. **For** :math:`i = 1` to ``monte_carlo_iters`` (e.g., 2000):

   * Generate a random assignment vector :math:`A` respecting :math:`P_G, P_L, P_C`.
   * Calculate Link Ratios (:math:`R_{links}`) for the 6 types: GG, LL, CC, GL, GC, LC.
   * Calculate Entropy :math:`W` using the entropy function:
   
     .. math::
     
        W = \text{EntropyFunc}(R_{bus}, R_{links})

   * Add :math:`W` to ``samples``.

3. Calculate Mean (:math:`\mu_W`) and Standard Deviation (:math:`\sigma_W`) of ``samples``.

4. Calculate Distance Parameter :math:`d` based on :math:`\log(N)` and reference curves.

5. **Compute Target:** 

.. math::
   
      W^* = (d \times \sigma_W) + \mu_W

6. **Set Convergence Criteria:** 

.. math::
   
      \epsilon = \text{function}(\sigma_W, N)

AIS Optimization Loop
---------------------

The algorithm mimics the immune system's Clonal Selection Principle: best antibodies (solutions) proliferate (clone) and undergo affinity maturation (mutation).

**Initialization:**

* Generate initial ``Population`` of :math:`K` random assignments.

**Loop** (until ``max_iter`` or ``BestError`` < :math:`\epsilon`):

1. **Affinity Evaluation:**

   * For each individual :math:`S` in ``Population``:
   
     * Calculate its Entropy :math:`W_S`.
     * Calculate **Error** (Affinity):
     
       .. math::
       
          \text{Error} = |W^* - W_S|

   * Sort ``Population`` by Error (Ascending).
   * Update ``BestSolution`` and ``BestError``.

2. **Convergence Check:**

   * **If** ``BestError`` < :math:`\epsilon`, **Break**.

3. **Clonal Selection:**

   * Select the top 50% of ``Population`` (Elite pool).
   * **For** each individual :math:`S` with rank :math:`r` in Elite pool:
   
     * Calculate clone count (*Better rank* :math:`\to` *More clones*):
     
       .. math::
       
          N_c = \text{round}\left( \frac{\beta \cdot \text{scale}}{r} \right)

     * Create :math:`N_c` copies of :math:`S`.

4. **Hypermutation:**

   * **For** each clone :math:`C` derived from parent with rank :math:`r`:
   
     * Determine mutation intensity :math:`M_{rate}` proportional to rank :math:`r`. (*Worse parents* :math:`\to` *Higher mutation intensity*)
     * Perform mutation :math:`M_{rate}` times:
     
       * Select random node :math:`u`.
       * Flip type of :math:`u` to random valid type :math:`\{Gen, Load, Conn\}`.

   * Collect all mutated clones into ``ClonePool``.

5. **Receptor Editing (Diversity Injection):**

   * Generate 10% new random solutions (``FreshPool``) to prevent local optima.

6. **Selection (Next Generation):**

   * Define new pool:
   
     .. math::
     
        \text{CombinedPool} = \text{FreshPool} \cup \text{ClonePool} \cup \text{ElitePool}

   * Evaluate Error for all in ``CombinedPool``.
   * Sort and truncate ``CombinedPool`` to original size :math:`K`.
   * Set ``Population`` = Truncated Pool.

Final Output
------------

* Return ``BestSolution`` converted to a dictionary ``{NodeID: BusType}``.