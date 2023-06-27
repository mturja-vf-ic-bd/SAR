# Partition the graph
# python3 examples/partition_arxiv_products.py --dataset-name ogbn-arxiv --num-partitions 2

for p in 10 20 30 50 100 200
do
   python3 examples/partition_arxiv_products.py --dataset-name ogbn-arxiv --num-partitions $p
done
