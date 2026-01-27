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


Preliminaries
^^^^^^^^^^^^^
For a graph $G=(V,E)$ with $n$ nodes, we denote the degree of a node $i$ as $d_i$, and the degree sequence as $\mathbf{d}=[d_1,d_2,\dots,d_n]$. 
Some graph metrics. 

* average distance in a connected graph is the average distance between all pairs of vertices 

.. math:: \text{avgDist} = \frac{1}{n(n-1)} \sum_{i\in V} \sum_{j\in V, j\neq i} d(i,j)

* diameter $\delta$ of a connected graph is the maximum distance over all pairs of nodes 

.. math:: \delta = \max_{i,j\in V, i\neq j} d(i,j)

* local clustering coefficient (LCC) of a vertex is to measure how tightly connected the neighborhood of a vertex is: 

.. math:: lcc(i) = \frac{\# \text{triangles incident to vertex } i}{d_i(d_i -1)/2}

The LCC of a degree-1 vertex is undefined, and the local clustering coefficient of a graph is the average LCC over all vertices for which the LCC is defined. 

Power grid as network-of-networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



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




Synthetic generation of topology generation inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Diameter $\delta$: The diameter of same-voltage subgraphs is on the order of $\sqrt{n}$, also suggested by `Young et al. (2018) <https://ieeexplore.ieee.org/document/8586475>`_. E.g., one can use the approximate function of the form $f(n) = c\cdot n^k$ to estimate the diameter. In this work, it yields $c\approx 1.301$ and $k\approx 0.574$ using the three sets of data. 


* Same-voltage degree sequence $\mathbf{d}^X$: It was suggested to use the following 

  * a generalized log-normal degree distribution, where the number $n_d$ of degree $d$ nodes follows
      
      .. math:: n_d \propto \exp\Big(-\bigg(\frac{\log d}{\alpha}\bigg)^\beta\Big)
      
      for some parameters $\alpha, \beta$. To optimize $\alpha, \beta$, one needs user-specified target values for average degree $\bar{d}$ and maximum degree $d_\max$. This work finds a consistent average degree across same-voltage subgraphs, with mean $\bar{d}=2.425$ and standard deviation $0.1846$ and coefficient of variation $7.6\%$. 

  * a power law (a scale-free network) where the degree distribution follows a power law

      .. math:: n_d \propto d^{-\gamma}
      
      for some parameter $\gamma$. To optimizer $\gamma$, the maximum degree $d_\max$ is required, which is suggested $d_\max \sim n^{1/\gamma'}$ with some $\gamma'$. Fitting $d_\max$ in the three sets of data of this work using $g(n)=c\cdot n^{1/4}$ yields $c\approx 1.517$.

  Given the required parameter(s), average and/or maximum degree, we can use `Kolda et al. (2014) <https://arxiv.org/abs/1302.6636>`_ to optimize the distribution parameters and obtain the degree sequences. 
  
* Transformer degree sequence $\mathbf{t}[X,Y]$: For any given transformer subgraph between voltage levels $X_i$ and $X_j$, there are two related transformer degree sequences $\mathbf{t}[X_i, X_j]$ and $\mathbf{t}[X_j,X_i].$ The first step to synthetically generating these, one needs to specify how many vertices participate in each transformer graph, i.e., how many vertices of voltage $X_i$ are incident to a transformer edge having the other endpoint in voltage $X_j$, and vice versa for $X_j$. This number can also be approximated based on the number of vertices in the same-voltage subgraphs. 

Given a set of $k$ voltage levels $\mathcal{X}=\{X_1,\dots,X_k\}$, denote the number of voltage $X_i$ vertices by $n_i$, and the number of $X_i$ vertices that are incident via a transformer to a voltage $X_j$ vertex by $t_i^j$. A function $h(n_i,n_j)\approx t_i^j$ of the following form 

.. math:: h(n_i,n_j) = c \cdot \min \{n_i,n_j\}

is used to approximate this number with an optimal $c\approx 0.174$. 

Then, since transformer subgraphs are relatively small graphs consisting almost entirely of disjoint, small $k$-stars, where the number of $k$-stars decreases exponentially in $k$, their degree distributions are short and steep. Thus, one can consider simple power-law degree distributions for some large power-law exponent $\gamma$. An optimal $\gamma\approx4.5$ is found by fitting the data. 

Finally, for each pair of voltage levels $(X_i,X_j)$, one may draw a single power law degree distribution on $h(n_i,n_j)$ vertices. Note that since transformer subgraphs are bipartitie graphs, the degree sequences $\mathbf{t}[X_i, X_j]$ and $\mathbf{t}[X_j,X_i]$ must sum to the same value for each pair of voltage levels (a necessary condition). This constraint can be achieved by iteratively modifying the outputted degree sequences --- nullifying the degree of randomly chosen vertices from the larger-sum sequence while redrawing degrees for null-degreed vertices in the smaller-sum sequence, until the sums are equal. 