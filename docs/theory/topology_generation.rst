########################################
Chung-Lu-Chain grid topology generation
########################################

Here we explain the graph model, Chung-Lu-Chain model, used to generate the raw grid topology. 

`Aksoy et al. (2018) <https://academic.oup.com/comnet/article-abstract/7/1/128/5073058?redirectedFrom=fulltext>`_ investigated the topological properties of power grids from the US Eastern Internconnection, Texas Interconnection, and Poland transmission system power grids and find 
* subgraphs induced by nodes of the same voltage level exhibit shared structural properties **atypical to small-world networks**, including 

  * low local clustering, 
  * large diameter, and 
  * large average distance 

* subgraphs induced by transformer edges linking nodes of different voltage types contain **mainly small, disjoint star graphs**. 

The method proposed by the authors includes two phases:

1. the first phase uses the **Chung-Lu random graph model**, taking desired node degrees and desired diameter as inputs, and outputs the subgraphs of same-voltage level; and 
2. the second phase uses a simple **random start graph model** to connect the subgraphs of different voltage levels. 