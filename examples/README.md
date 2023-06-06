## Graph partitioning

``partition_arxiv_products.py`` partitions the ogbn-arxiv and ogbn-products  graphs from the [Open Graph Benchmarks](https://ogb.stanford.edu/) using DGL's metis-based partitioning. The general technique there can be used to partition arbitrary homogeneous graphs. Note that all node-related information must be included in the graph's ``ndata`` dictionary so that they are correctly partitioned with the graph. Similarly, edge-related information must be included in the graph's ``edata`` dictionary

``partition_mag.py`` partitions the [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) heterogeneous graph. Again, all node-related information are included in the graph's ``ndata`` for the relevant node types

## Full-batch Training

The script ``train_homogeneous_graph_basic.py`` demonstrates the basic functionality of SAR. It runs distriobuted training using a 3-layer GraphSage network on a partiotioned graph. If you want to train using ``N`` workers, then you need to launch the script ``N`` times, preferably on separate machines. For example, for ``N=2``, and assuming the two workers are on the same network file system, you can launch the 2 workers using the following two commands:

```shell
python3 train_homogeneous_graph_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2
python3 train_homogeneous_graph_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 1 --world-size 2

```
The worker with ``rank=0`` (the master)  will write its address to the file specified by the ``--ip-file`` option and the other worker(s) will read this file and connect to the master.


The ``train_homogeneous_graph_advanced.py`` script demonstrates more advanced features of SAR such as distributed construction of Message Flow Graphs (MFGs), and the multiple training modes supported by SAR. 

The ``train_heterogeneous_graph.py`` script demonstrates training on a heterogeneous graph (ogbn-mag). The script trains a 3-layer R-GCN.


## Sampling-based Training
The script ``train_homogeneous_sampling_basic.py`` demonstrates distributed sampling-based training on SAR. It demonstrates the unique ability of  the SAR library to run distributed sampling-based training followed by memory-efficient distributed full-graph inference. The script uses a 3-layer GraphSage network. Assuming 2 machines, you can launch training on the 2 machines using the following 2 commands, where each is executed on a different machine:

```shell
python3 train_homogeneous_sampling_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2
python3 train_homogeneous_sampling_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 1 --world-size 2

```

## Training with compression
There are 3 supported compression types : ``feature``, ``variable``, and ``pca``. They are supported in both full-graph training and sampling-based training modes. For full-graph training, you need to use the ``one_shot_aggregation`` training mode. For sampling-based training, you need to use the ``--hybrid-partitioning`` flag.

An example of full-graph training with compression

```shell
python3 train_homogeneous_graph_advanced.py --ip-file ./ip_file_papers8_comp_v --train-mode one_shot_aggregation --rank 0 --world-size 2 --train-iters 100 --fed_agg_round 1 --partitioning-json-file /lfs/lfs15/hmostafa/graph_partitions/ogbn-papers100M/split2/ogb-paper100M.json --cpu-run --comp_ratio 2 --min_comp_ratio 1 --max_comp_ratio 32 --compression_type variable --enable_cr  --train-compressor
python3 train_homogeneous_graph_advanced.py --ip-file ./ip_file_papers8_comp_v --train-mode one_shot_aggregation --rank 1 --world-size 2 --train-iters 100 --fed_agg_round 1 --partitioning-json-file /lfs/lfs15/hmostafa/graph_partitions/ogbn-papers100M/split2/ogb-paper100M.json --cpu-run --comp_ratio 2 --min_comp_ratio 1 --max_comp_ratio 32 --compression_type variable --enable_cr  --train-compressor
```



An example of sampling-based training with compression


```shell
python3 train_homogeneous_sampling_basic.py --ip-file ./ip_file8_prods8_sampling_hybrid_inplace_compression16 --rank 0 --world-size 2   --train-iters 50 --partitioning-json-file /lfs/lfs15/hmostafa/graph_partitions/ogbn-products_random/split2/ogb-product.json  --lr 0.01  --batch-size 1000 --num-workers 0 --cpu-run --backend ccl --hybrid-partitioning --max-collective-size 30000000  --comp_ratio 16 --compression_type feature --enable_cr
python3 train_homogeneous_sampling_basic.py --ip-file ./ip_file8_prods8_sampling_hybrid_inplace_compression16 --rank 1 --world-size 2   --train-iters 50 --partitioning-json-file /lfs/lfs15/hmostafa/graph_partitions/ogbn-products_random/split2/ogb-product.json  --lr 0.01  --batch-size 1000 --num-workers 0 --cpu-run --backend ccl --hybrid-partitioning --max-collective-size 30000000  --comp_ratio 16 --compression_type feature --enable_cr

```
