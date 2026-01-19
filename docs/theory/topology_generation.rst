########################################
Chung-Lu-Chain grid topology generation
########################################

Here we explain the graph model, Chung-Lu-Chain model, used to generate the raw grid topology. 
The methodology is mainly based on `Aksoy et al. (2018) <https://academic.oup.com/comnet/article-abstract/7/1/128/5073058?redirectedFrom=fulltext>`_, which investigated the topological properties of power grids from the US Eastern Internconnection, Texas Interconnection, and Poland transmission system power grids and found

1. subgraphs induced by nodes of the same voltage level exhibit shared structural properties **atypical to small-world networks**, including 

  * low local clustering, 
  * large diameter, and 
  * large average distance 

2. subgraphs induced by transformer edges linking nodes of different voltage types contain **mainly small, disjoint star graphs**. 

The method proposed by the authors includes two phases:

1. the first phase uses the **Chung-Lu random graph model**, taking desired node degrees and desired diameter as inputs, and outputs the subgraphs of same-voltage level; and 
2. the second phase uses a simple **random start graph model** to connect the subgraphs of different voltage levels. 





What is a Chung-Lu model? 
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Chung-Lu random graph model takes as an input a desired node degree sequence $\mathbf{d}=(d_1,\dots,d_n)$, with $d_i$ the desired degree of node $i$.
For every pair of nodes $i,j$, the probability of an edge $(i,j)$ is 

.. math:: \text{Pr}(\{i,j\} \text{ is an edge}) = \frac{d_i d_j}{2m}

where $m=\frac{1}{2}\sum_{i} d_i$ is the desired number of edges. 
In order to ensure that this probability is at most 1, one further requires that $\max_i d_i^2 \leq \sum_{k} d_k$. In expectation, each node achieves its desired degree $d_i$

.. math:: \mathbb{E}(\text{degree of node } i) = \sum_{i} \text{Pr} (\{i,j\} \text{ is an edge}) = \sum_j \frac{d_i d_j}{2m} = d_i.


Pseudo-algorithm
________________

Fast Chung-Lu model chooses the endpoints of $m$ edges by sampling these endpoints proportionally to their desired degree:

.. math:: \text{Pr}(i \text{ selected}) = \frac{d_i}{2m}.

The expected degree of a node is again its desired degree

.. math:: \text{Pr}(\{i,j\} \text{ is an edge}) = 2m \cdot \text{Pr}(i \text{ selected}) \cdot \text{Pr}(j \text{ selected}) = \frac{d_i d_j}{2m}.


Some prior work
^^^^^^^^^^^^^^^
Hines et al. (2010) considered the aggregate power grid graph from IEEE 300-bus test case and the US Eastern Interconnection grid, and found that the structure of these graphs differ substantially from the comparably sized random graph models like Erdõs-Rényi, preferential attachment, and small-world models in terms of the degree distribution, clustering, diameter and assortativity. 



Validation metrics
^^^^^^^^^^^^^^^^^^

`Birchfield et al. (2017) <https://www.mdpi.com/1996-1073/10/8/1233>`_ and `Birchfield et al. (2016) <https://ieeexplore.ieee.org/document/7725528>`_ build an extensive list of 18 validation metrics for assessing the realism of systhetic power grid data. 

* metrics of system proportions: substations, load and generation

  * number of buses per substation
  * substation voltage levels
  * percentage of substations containing load 
  * load at each bus 
  * ratio of total generation capacity to total load 
  * percent of substations containing generation 
  * capacities of generators 
  * percent of geneartors committed 
  * generator dispatch percentage 
  * generator reactive power limits 

* metrics of system network: transformers and transmission lines

  * transformer per-unit reactance: $\geq 80\%$ of transformers have a reactance value in the range $[0.05, 0.2]$, and the distribution is roughly normal, centered around 0.12, with some variation.
  * transformer MVA limit and $X/R$ ratio 
  * transmission line reactance 
  * transmission line $X/R$ ratio and MVA limit 
  * ratio of transmission lines to substations, at a single nominal voltage level 
  * percent of lines on the minimum spanning tree 
  * distance of transmission lines along the Delaunay triangulation 
  * topology-related graph theory statistics: 

    * distribution of nodal degrees, clustering coefficient, average shortest path length 

  * ratio of total lengths of all lines to the length of the minimum spanning tree


