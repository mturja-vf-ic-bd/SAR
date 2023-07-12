




part=200 # partitions
p=0
python3 examples/train_homogeneous_single_node.py --cpu-run --partitioning-json-file ogbn-arxiv.json --train-iters 100 --rank $p --backend ccl --world-size 1 --partitions $part &

