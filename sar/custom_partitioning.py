import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional
import os.path as osp
import numpy as np
from tqdm import tqdm
from .common_tuples import PartitionData, ShardEdgesAndFeatures

numpy_to_torch_dtype_dict = {
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32
}


def get_type_ordered_edges(edge_types: Tensor) -> Tensor:
    reordered_edge_mask: List[Tensor] = []
    n_edge_types = edge_types.max().item()
    for edge_type_idx in range(n_edge_types):
        edge_mask_typed = edge_types == edge_type_idx
        reordered_edge_mask.append(
            edge_mask_typed.nonzero(as_tuple=False).view(-1))

    return torch.cat(reordered_edge_mask)


def permute_and_split(indices, n_parts):
    assert indices.ndim == 1
    n_nodes = indices.size(0)
    indices = indices[torch.randperm(n_nodes)]

    k, m = divmod(n_nodes, n_parts)
    return [indices[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n_parts)]


def load_custom_partitioning(output_dir,
                             own_partition,
                             n_parts,
                             active_type_data):
    print(f'loading partition {own_partition} from {output_dir}')
    # active_type_data is something like {'feature',((type,type_shift,f_name),(type,type_shift,f_name)..)}
    all_shard_edges: List[ShardEdgesAndFeatures] = []
    n_parts_range = tqdm(range(n_parts))
    for src_part_idx in n_parts_range:
        n_parts_range.set_description(
            f'loading source partition {src_part_idx}')
        src_nodes, dst_nodes, edge_feature_dict = torch.load(
            osp.join(output_dir, f'part_{src_part_idx}_{own_partition}'))
        if '_TYPE' in edge_feature_dict:
            n_parts_range.set_description('reordering edges based on _TYPE')
            etype_reordering = get_type_ordered_edges(
                edge_feature_dict['_TYPE'])
            src_nodes = src_nodes[etype_reordering]
            dst_nodes = dst_nodes[etype_reordering]
            edge_feature_dict['_TYPE'] = edge_feature_dict['_TYPE'][etype_reordering]
        all_shard_edges.append(ShardEdgesAndFeatures(
            (src_nodes, dst_nodes),
            edge_feature_dict))

    print('loading partition original indices and feature data')
    orig_indices, node_features_dict, node_ranges, node_type_shifts = torch.load(
        osp.join(output_dir, f'part_{own_partition}'))

    type_data_prog = tqdm(active_type_data.items())
    for feature_name, type_data in type_data_prog:
        for type_id,  f_name in type_data:
            type_data_prog.set_description(
                f'loading feature {feature_name} for node type {type_id} from {f_name}')
            feat_data = np.load(f_name)  # , mmap_mode='r+')
            if type_id is None:
                node_features_dict[feature_name] = torch.from_numpy(
                    feat_data[orig_indices.numpy()])
            else:
                if feature_name not in node_features_dict:
                    shape = list(feat_data.shape)
                    shape[0] = node_ranges[own_partition][1] - \
                        node_ranges[own_partition][0]
                    node_features_dict[feature_name] = torch.zeros(shape,
                                                                   dtype=numpy_to_torch_dtype_dict[feat_data.dtype])

                node_types = node_features_dict['_TYPE']
                relevant_nodes = (node_types == type_id).nonzero().view(-1)
                # Loading features
                type_data_prog.set_description(
                    f'shifting indices of {feature_name} to homogeneous indices')
                type_shift = node_type_shifts[type_id]
                loaded_features = feat_data[orig_indices[relevant_nodes].numpy(
                ) - type_shift]
                node_features_dict[feature_name][relevant_nodes] = torch.from_numpy(
                    loaded_features)

    return PartitionData(all_shard_edges,
                         node_ranges,
                         node_features_dict,
                         [''],
                         ['']
                         )


def random_partition(graph, n_parts, output_dir, node_feature_dict, edge_feature_dict,
                     train_indices, val_indices, test_indices, node_type_shifts):
    print(f'random partitioning a graph with {graph.number_of_nodes()} \
    nodes and {graph.number_of_edges()} edges')

    print('obtaining unlabeled indices')
    train_test_val_indices = torch.cat(
        (train_indices, val_indices, test_indices))
    all_nodes = torch.arange(graph.number_of_nodes())
    unlabeled_indices = all_nodes[~torch.isin(
        all_nodes, train_test_val_indices)]
    del train_test_val_indices, all_nodes

    print('Splitting train/val/test/unlabeled nodes')
    (train_indices_split,
     val_indices_split,
     test_indices_split,
     unlabeled_indices_split) = map(lambda x: permute_and_split(x, n_parts),
                                    [train_indices,
                                    val_indices,
                                    test_indices,
                                    unlabeled_indices])

    del unlabeled_indices

    print('constructing nodes in each partition')
    part_nodes = [torch.cat((x, y, z, w)) for x, y, z, w in zip(train_indices_split,
                                                                val_indices_split,
                                                                test_indices_split,
                                                                unlabeled_indices_split)]

    del train_indices_split, val_indices_split, test_indices_split, unlabeled_indices_split

    print('constructing contiguous mapping array')
    mapping = torch.empty(graph.number_of_nodes(), dtype=torch.int64)
    start = 0
    node_ranges = []
    for p in part_nodes:
        mapping[p] = torch.arange(start, start + p.size(0))
        node_ranges.append((start, start + p.size(0)))
        start += p.size(0)

    assert node_ranges[-1][1] == graph.number_of_nodes()

    print('Mapping node indices in each partition to a contiguous range')
    src_nodes, dst_nodes = graph.all_edges()
    src_nodes = mapping[src_nodes]
    dst_nodes = mapping[dst_nodes]

    def in_range(tens, n_range):
        return torch.logical_and(tens >= n_range[0], tens < n_range[1])

    print(f'saving partition data to {output_dir}')
    n_parts_range = tqdm(range(n_parts))
    for dst_part_idx in n_parts_range:
        dst_part = in_range(dst_nodes, node_ranges[dst_part_idx])
        for src_part_idx in range(n_parts):
            src_dst_part = torch.logical_and(dst_part,
                                             in_range(src_nodes, node_ranges[src_part_idx]))
            src_dst_part = src_dst_part.nonzero().view(-1)

            partition_edge_feature_dict = {}
            for k, v in edge_feature_dict.items():
                partition_edge_feature_dict[k] = v[src_dst_part]

            n_parts_range.set_description(
                f'saving  edge data for shard {src_part_idx} -> {dst_part_idx}')
            torch.save((src_nodes[src_dst_part],
                        dst_nodes[src_dst_part],
                        partition_edge_feature_dict),
                       osp.join(output_dir, f'part_{src_part_idx}_{dst_part_idx}'))

        partition_node_feature_dict = {}
        for k, v in node_feature_dict.items():
            partition_node_feature_dict[k] = v[part_nodes[dst_part_idx]]

        n_parts_range.set_description(
            f'saving node feature data for partition {dst_part_idx}')
        torch.save((part_nodes[dst_part_idx],
                    partition_node_feature_dict,
                    node_ranges, node_type_shifts), osp.join(output_dir, f'part_{dst_part_idx}'))
