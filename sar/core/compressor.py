import enum
from math import ceil, floor, log2
from typing import List, Tuple
import logging
from xmlrpc.client import boolean
import dgl
from collections.abc import MutableMapping
from numpy import append
import torch
import math
from torch import seed
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sar.comm import exchange_tensors, rank
from sar.config import Config


logger = logging.getLogger(__name__)


class CompressorDecompressorBase(nn.Module):
    '''
    Base class for all communication compression modules
    '''

    def __init__(
            self):
        super().__init__()

    def compress(self, tensors_l: List[Tensor]):
        '''
        Take a list of tensors and return a list of compressed tensors
        '''
        return tensors_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''
        return channel_feat


class FeatureCompressorDecompressor(CompressorDecompressorBase):
    def __init__(self, feature_dim: List[int], comp_ratio: List[float]):
        """
        A feature-based compression decompression module. The compressor compresses outgoing
        tensor along feature dimension and decompresses it back to original size on the receiving
        client side. The model follows a autoencoder like architecture where sending client uses the
        encoder and receiving client uses the decoder.

        :param feature_dim: A list of feature dimension for each layer of GNN including input layer.
        :type feature_dim: List[int]
        :param comp_ratio: A list of compression ratio for each layer of GNN to allow different
        compression ratio for different layers.
        :type comp_ratio: List[float]
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.compressors = nn.ModuleDict()
        self.decompressors = nn.ModuleDict()
        for i, f in enumerate(feature_dim):
            k = floor(f/comp_ratio[Config.current_layer_index])
            print(i, k, f, comp_ratio[Config.current_layer_index])
            self.compressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, k),
                nn.ReLU()
            )
            self.decompressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(k, f)
            )

    def compress(self, tensors_l: List[Tensor], iter: int = 0, scorer_type=None):
        '''
        Take a list of tensors and return a list of compressed tensors

        :param iter: Ignore. Added for compatiability.
        :param enable_vcr: Ignore. Added for compatiability.
        :param scorer_type: Ignore, Added for compatiability.
        '''
        # Send data to each client using same compression module
        logger.debug(
            f"index: {Config.current_layer_index}, tensor_sz: {tensors_l[0].shape}")
        tensors_l = [self.compressors[f"layer_{Config.current_layer_index}"](val)
                     if Config.current_layer_index < Config.total_layers - 1
                     else val for val in tensors_l]
        return tensors_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''
        decompressed_tensors = [self.decompressors[f"layer_{Config.current_layer_index}"](c)
                                if Config.current_layer_index < Config.total_layers - 1
                                else c for c in channel_feat]
        return decompressed_tensors


class VariableFeatureCompressorDecompressor(CompressorDecompressorBase):
    def __init__(self, feature_dim: List[int], slope: int,max_comp_ratio: float, min_comp_ratio: float):
        """
        A feature-based compression decompression module. The compressor compresses outgoing
        tensor along feature dimension and decompresses it back to original size on the receiving
        client side. The model follows a autoencoder like architecture where sending client uses the
        encoder and receiving client uses the decoder.

        :param feature_dim: A list of feature dimension for each layer of GNN including input layer.
        :type feature_dim: List[int]
        :param comp_ratio: A list of compression ratio for each layer of GNN to allow different
        compression ratio for different layers.
        :type comp_ratio: List[float]
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.compressors = nn.ModuleDict()
        self.decompressors = nn.ModuleDict()
        self.slope = slope
        self.min_comp_ratio = min_comp_ratio
        self.max_comp_ratio = max_comp_ratio
        


        for i, f in enumerate(feature_dim):
            Config.current_layer_index = i
            self.compressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, f),
            )
            self.decompressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, f)
            )
            compressed_size = self.get_compressed_size(0)
            self.decompressors[f"layer_{i}"][0].weight[:,
                                                       compressed_size:].data.fill_(0)
            print(f'set decompressor weight with size {self.decompressors[f"layer_{i}"][0].weight.size()} \
            from {compressed_size} to end with zero')

    def get_compressed_size(self, iter):

        slope = (self.max_comp_ratio - self.min_comp_ratio) / \
            (Config.total_train_iter-2)
        # current_ratio = self.max_comp_ratio - iter * slope
        current_ratio = self.max_comp_ratio - self.slope * iter * slope

#        current_ratio = self.max_comp_ratio * math.exp(-(iter * math.log(
#            self.max_comp_ratio/self.min_comp_ratio))/(Config.total_train_iter - 2))

        current_ratio = max(current_ratio, self.min_comp_ratio)

        active_compressor = self.compressors[f"layer_{Config.current_layer_index}"]
        active_feature_dim = active_compressor[0].weight.size(0)

        if Config.current_layer_index < Config.total_layers - 1:
            compressed_feature_dim = int(
                active_feature_dim * 1.0 / current_ratio)

        else:
            compressed_feature_dim = active_feature_dim
        print(
            f'compressed feature dim at layer {Config.current_layer_index} : {compressed_feature_dim}', 'iter',iter,current_ratio) #juan added this
        return compressed_feature_dim

    def compress(self, tensors_l: List[Tensor], iter: int = 0, scorer_type=None):
        '''
        Take a list of tensors and return a list of compressed tensors

        :param iter: Ignore. Added for compatiability.
        :param enable_vcr: Ignore. Added for compatiability.
        :param scorer_type: Ignore, Added for compatiability.
        '''
        # Send data to each client using same compression module
        logger.debug(
            f"index: {Config.current_layer_index}, tensor_sz: {tensors_l[0].shape}")
        active_compressor = self.compressors[f"layer_{Config.current_layer_index}"]
        active_feature_dim = active_compressor[0].weight.size(0)
        compressed_feature_dim = self.get_compressed_size(iter)

        result_l = []
        for val in tensors_l:
            if Config.current_layer_index < Config.total_layers - 1:
                res = active_compressor(val)
                res = res[:, :compressed_feature_dim].contiguous()
            else:
                res = val
            result_l.append(res)
        return result_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''

        active_decompressor = self.decompressors[f"layer_{Config.current_layer_index}"]
        active_feature_dim = active_decompressor[0].weight.size(0)

        result_l = []
        for val in channel_feat:
            if Config.current_layer_index < Config.total_layers - 1:
                compressed_feature_dim = val.size(-1)
                val = F.pad(
                    val, (0, active_feature_dim - compressed_feature_dim))

                res = active_decompressor(val)
            else:
                res = val
            result_l.append(res)
        return result_l


class PCACompressorDecompressor(CompressorDecompressorBase):
    def __init__(self, feature_dim: List[int], max_comp_ratio: float, min_comp_ratio: float):
        """
        A feature-based compression decompression module. The compressor compresses outgoing
        tensor along feature dimension and decompresses it back to original size on the receiving
        client side. The model follows a autoencoder like architecture where sending client uses the
        encoder and receiving client uses the decoder.

        :param feature_dim: A list of feature dimension for each layer of GNN including input layer.
        :type feature_dim: List[int]
        :param comp_ratio: A list of compression ratio for each layer of GNN to allow different
        compression ratio for different layers.
        :type comp_ratio: List[float]
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.min_comp_ratio = min_comp_ratio
        self.max_comp_ratio = max_comp_ratio

    def get_compressed_size(self, iter):

        slope = (self.max_comp_ratio - self.min_comp_ratio) / \
            (Config.total_train_iter-2)
        current_ratio = self.max_comp_ratio - iter * slope

#        current_ratio = self.max_comp_ratio * math.exp(-(iter * math.log(
#            self.max_comp_ratio/self.min_comp_ratio))/(Config.total_train_iter - 2))

        current_ratio = max(current_ratio, self.min_comp_ratio)
        if current_ratio < 2.5:
            return None

        active_feature_dim = self.feature_dim[Config.current_layer_index]

        if Config.current_layer_index < Config.total_layers - 1:
            compressed_feature_dim = int(
                active_feature_dim * 1.0 / current_ratio)

        else:
            compressed_feature_dim = active_feature_dim
        print(
            f'compressed feature dim at layer {Config.current_layer_index} : {compressed_feature_dim}')
        return compressed_feature_dim

    def compress(self, tensors_l: List[Tensor], iter: int = 0, scorer_type=None):
        '''
        Take a list of tensors and return a list of compressed tensors

        :param iter: Ignore. Added for compatiability.
        :param enable_vcr: Ignore. Added for compatiability.
        :param scorer_type: Ignore, Added for compatiability.
        '''

        compressed_feature_dim = self.get_compressed_size(iter)

        result_l = []
        for val in tensors_l:
            if compressed_feature_dim is not None and Config.current_layer_index < Config.total_layers - 1 and val.numel() > 0:
                U, S, V = torch.pca_lowrank(
                    val, compressed_feature_dim, center=False, niter=1)
                res = torch.cat((U, S.unsqueeze(0), V))
            else:
                res = val
            result_l.append(res)
        return result_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''

        active_feature_dim = self.feature_dim[Config.current_layer_index]

        result_l = []
        for val in channel_feat:
            if Config.current_layer_index < Config.total_layers - 1 and val.numel() > 0:
                U, S, V = (val[:-active_feature_dim-1],
                           val[-active_feature_dim-1:-active_feature_dim],
                           val[-active_feature_dim:])
                res = (U * S) @ V.T
            else:
                res = val
            result_l.append(res)
        return result_l


def compute_CR_exp(step, iter):
    # Decrease CR from 2**p to 2**1
    return 2**(ceil((Config.total_train_iter - iter)/step))


def compute_CR_linear(init_CR, slope, step, iter):
    # Decrease CR using b - a*x
    return init_CR - slope * (iter // step)


def pool(val, score, k):
    print('pool', val, score, k)
    sel_scores, idx = torch.topk(score, k=k, dim=0)
    idx = idx.squeeze(-1)
    new_val = torch.mul(val[idx, :], sel_scores)
    return new_val, idx


class NodeCompressorDecompressor(CompressorDecompressorBase):
    """
    A node-based compression decompression module. The sending client selects a subset of nodes
    that it needs to send and the receiving client replaces the missing nodes with 0. It consists 
    of a ranking module which ranks the node based on their feature using a one-layer neural network.
    Then it selects a fraction of the nodes based on compression ratio. The compression ratio can be
    fixed or variable over training iterations. This whole ranking and selection process is similar
    to pooling operator in Graph-UNet (https://github.com/HongyangGao/Graph-U-Nets/blob/master/src/utils/ops.py#L64)

    :param feature_dim: A list of feature dimension for each layer of GNN including input layer.
    :type feature_dim: List[int]
    :param comp_ratio: A list of compression ratio for each layer of GNN to allow different
    compression ratio for different layers.
    :type comp_ratio: float
    """

    def __init__(
            self,
            feature_dim: List[int],
            comp_ratio: float,
            step: int,
            enable_vcr: bool):

        super().__init__()
        self.feature_dim = feature_dim
        self.scorer = nn.ModuleDict()
        self.comp_ratio = comp_ratio
        self.step = step
        self.enable_vcr = enable_vcr
        print('casting class', feature_dim, comp_ratio, enable_vcr)
        for i, f in enumerate(feature_dim):
            self.scorer[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, 1),
                nn.Sigmoid()
            )

    def compress(
            self,
            tensors_l: List[Tensor],
            iter: int = 0,
            scorer_type: str = "learnable"):
        """
        Take a list of tensors and return a list of compressed tensors

        :param tensors_l: List of send tensors for each graph shard
        :type List[Tensor]
        :param iter: The training iteration number
        :type int
        :param step: Number of steps for which CR is constant
        :type int
        :param enable_vcr: Enable variable compression ratio
        :type str
        :param scorer_type: Module by which the nodes will be ranked before sending. 
        There are two possible types: "learnable" / "random". In learnable scorer, the
        scores will be computed by one-layer neural network based on features of the nodes.
        In case of "random", the nodes will be selected randomly. Default: "learnable"
        :type str
        """

        compressed_tensors_l = []
        sel_indices = []

        # Compute compression ratio
        if self.enable_vcr:
            comp_ratio = compute_CR_exp(self.step, iter)
        else:
            assert self.comp_ratio is not None, \
                "Compression ratio can't be None for fixed compression ratio"
            comp_ratio = self.comp_ratio
        comp_ratio = max(1, comp_ratio)

        for val in tensors_l:
            # Compute ranking scores
            if scorer_type == "learnable":
                score = self.scorer[f"layer_{Config.current_layer_index}"](val)
            elif scorer_type == "random":
                score = torch.rand(val.shape[0], 1)
                score = torch.sigmoid(score)
            else:
                raise NotImplementedError(
                    "Scorer type should be either learnable or random")
            k = val.shape[0] // comp_ratio
            # Send at least 1 node if CR is too high for compatiability
            k = max(1, k)
            print('in for', val, score, k)
            new_val, idx = pool(val, score, k)
            compressed_tensors_l.append(new_val)
            sel_indices.append(idx)

        return compressed_tensors_l, sel_indices

    def decompress(
            self,
            args):
        '''
        Decompress received tensors by creating properly shaped recv tensors
        '''

        channel_feat = args[0]
        sel_indices = args[1]
        sizes_expected_from_others = args[2]

        decompressed_tensors_l = []

        for i in range(len(sizes_expected_from_others)):
            new_val = channel_feat[i].new_zeros(
                sizes_expected_from_others[i],
                channel_feat[i].shape[1])
            new_val[sel_indices[i], :] = channel_feat[i]
            decompressed_tensors_l.append(new_val)

        return decompressed_tensors_l


class SubgraphCompressorDecompressor(CompressorDecompressorBase):
    """
    A class to perform the subgraph-based compression decompression mechanism.
    While sending a set of node features to remote clients this class sends a 
    representation of the subgraph induced by those nodes. This representation
    is better both in terms of privacy (since it's not a node-specific representation) 
    and communication overhead (since it's a compressed representation). To learn this
    representation, the compressor first passes the node features through a GNN using the 
    induced subgraph. Then it uses a ranking module to pool a subset of node representation 
    and sends them. This is only applied to the raw features (layer=0) of the nodes and 
    not on the hidden representation of the GNN. 
    Upon receiving, the remote client diffuses these representation using 
    the induced subgraph structure.

    :param feature_dim: List of integers representing the feature dimensions at each GNN layer.
    :type feature_dim: List[int]
    :param full_local_graph: A graph representing all the edges incoming to this partition.
    :type full_local_graph: DistributedBlock
    :param indices_required_from_me: The local node indices required by every other partition to \
        carry out one-hop aggregation
    :type indices_required_from_me: List[Tensor]
    :param tgt_node_range: Node ranges for target clients.
    :type tgt_node_range: Tuple[int, int]
    :param comp_ratio: Fixed compression ratio. n_nodes/comp_ratio nodes will be sent.
    :type comp_ratio: float
    """

    def __init__(
        self,
        feature_dim: List[int],
        step: int,
        enable_vcr: bool,
        full_local_graph,
        indices_required_from_me: List[Tensor],
        tgt_node_range: Tuple[int, int],
        comp_ratio: float
    ):
        super().__init__()
        self.full_local_graph = full_local_graph
        self.tgt_node_range = tgt_node_range
        self.comp_ratio = comp_ratio
        self.step = step
        self.enable_vcr = enable_vcr

        # Create learnable modules
        self.pack = nn.ModuleDict()     # GCN module to aggregate information
        self.scorer = nn.ModuleDict()   # Ranking module
        self.unpack = nn.ModuleDict()   # GCN module to diffuse information

        for i, f in enumerate(feature_dim):
            layer = f"layer_{i}"
            self.pack[layer] = nn.ModuleList([
                dgl.nn.SAGEConv(f, f, aggregator_type='mean')]
            )
            self.scorer[layer] = nn.Sequential(
                nn.Linear(f, 1),
                nn.Sigmoid()
            )
            self.unpack[layer] = nn.ModuleList([
                SageConvExt(f, f, update_func="diffuse"),
                SageConvExt(f, f, update_func="diffuse")]
            )
        # Create induced subgraphs for source nodes
        self.induced_boundary_graphs = []
        self.edges_src_nodes = []
        self.edges_tgt_nodes = []
        self.remote_boundary_graphs = []
        for i, local_src_nodes in enumerate(indices_required_from_me):
            boundary_induced_subgraph, edges_src_nodes, edges_tgt_nodes =\
                self._construct_boundary_subgraph(local_src_nodes)
            self.induced_boundary_graphs.append(boundary_induced_subgraph)
            self.edges_src_nodes.append(edges_src_nodes)
            self.edges_tgt_nodes.append(edges_tgt_nodes)

    def _construct_boundary_subgraph(self, seed_nodes: Tensor):
        assert torch.all(seed_nodes == torch.sort(seed_nodes)[0]), \
            "seed_nodes should be sorted"
        # Select edges where source and target nodes are the local boundary nodes (seed nodes)
        # First convert node id's to global id's
        graph_to_global = torch.cat(self.full_local_graph.unique_src_nodes)
        global_tgt_nodes = seed_nodes + self.tgt_node_range[0]
        global_src_nodes = global_tgt_nodes  # src and tgt are the same
        global_tgt_edges = self.full_local_graph.all_edges()[
            1] + self.tgt_node_range[0]
        global_src_edges = graph_to_global[self.full_local_graph.all_edges()[
            0]]

        # Find edges where source and target nodes are boundary nodes
        induced_edge_locs_tgt = torch.isin(global_tgt_edges, global_tgt_nodes)
        induced_edge_locs_src = torch.isin(global_src_edges, global_src_nodes)
        induced_edge_locs = torch.logical_and(
            induced_edge_locs_tgt, induced_edge_locs_src)
        assert induced_edge_locs.size(0) > 0, \
            "There should be at least one edge in the induced graph"

        # Convert to graph id
        src_nodes = global_src_edges[induced_edge_locs]
        tgt_nodes = global_tgt_edges[induced_edge_locs]
        unique_src_nodes, unique_src_nodes_inverse = \
            torch.unique(src_nodes, sorted=True, return_inverse=True)
        unique_tgt_nodes, unique_tgt_nodes_inverse = \
            torch.unique(tgt_nodes, sorted=True, return_inverse=True)
        assert torch.all(unique_src_nodes == unique_tgt_nodes), \
            "Source and Target nodes must be the same"
        edges_src_nodes = torch.arange(unique_src_nodes.size(0))[
            unique_src_nodes_inverse]
        edges_tgt_nodes = torch.arange(unique_tgt_nodes.size(0))[
            unique_tgt_nodes_inverse]

        induced_graph = dgl.graph(
            (edges_src_nodes, edges_tgt_nodes),
            num_nodes=len(seed_nodes),
            device=self.full_local_graph.device)

        return dgl.add_self_loop(induced_graph), edges_src_nodes, edges_tgt_nodes

    def compress(
            self,
            tensors_l: List[Tensor],
            iter: int = 0,
            scorer_type=None):
        """
        Take a list of tensors and return a list of compressed tensors

        :param tensors_l: List of send tensors for each graph shard
        :type List[Tensor]
        :param iter: The training iteration number
        :type int
        :param step: Number of steps for which CR is constant
        :type int
        :param enable_vcr: Enable variable compression ratio
        :type bool
        :param scorer_type: Ignore. Added for compatiability
        :type str
        """

        compressed_tensors_l = []
        sel_indices = []
        if self.enable_vcr:
            comp_ratio = compute_CR_exp(self.step, iter)
        else:
            assert self.comp_ratio is not None, \
                "Compression ratio can't be None for fixed compression ratio"
            comp_ratio = self.comp_ratio
        comp_ratio = max(1, comp_ratio)

        for i, val in enumerate(tensors_l):
            if Config.current_layer_index == 0:
                g = self.induced_boundary_graphs[i]
                net = self.pack[f"layer_{Config.current_layer_index}"]
                for j, conv in enumerate(net):
                    val = conv(g, val)
                    if j < len(net) - 1:
                        val = nn.ReLU()(val)
            score = self.scorer[f"layer_{Config.current_layer_index}"](val)
            k = val.shape[0] // comp_ratio
            k = max(1, k)  # Send at least 1 node if CR is too high.
            new_val, idx = pool(val, score, k)
            compressed_tensors_l.append(new_val)
            sel_indices.append(idx)
        return compressed_tensors_l, sel_indices, \
            self.edges_src_nodes, self.edges_tgt_nodes

    def decompress(
            self,
            args):
        '''
        Decompress received tensors by creating properly shaped recv tensors

        :param args: List of received messages. First entry of the list is the 
        node features received. Second entry is indices of the nodes that were sent.
        Third and fourth entries are the src and tgt of the induces subgraph and the last
        entry is the number of total nodes sent from each remote clients.
        :type args: List[List[Tensor]]
        '''

        channel_feat = args[0]
        sel_indices = args[1]
        edges_src_nodes = args[2]
        edges_tgt_nodes = args[3]
        sizes_expected_from_others = args[4]
        decompressed_tensors_l = []

        for i in range(len(sizes_expected_from_others)):
            new_val = channel_feat[i].new_zeros(
                sizes_expected_from_others[i],
                channel_feat[i].shape[1])
            new_val[sel_indices[i], :] = channel_feat[i]
            induced_graph = dgl.graph((edges_src_nodes[i], edges_tgt_nodes[i]))
            net = self.unpack[f"layer_{Config.current_layer_index}"]
            for idx, conv in enumerate(net):
                new_val = conv(induced_graph, new_val, sel_indices[i])
                if idx < len(net) - 1:
                    new_val = nn.ReLU()(new_val)
            decompressed_tensors_l.append(new_val)

        return decompressed_tensors_l
