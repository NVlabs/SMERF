import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv import ConfigDict
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmdet3d.models import NECKS, BACKBONES
from mmdet3d.models.builder import build_backbone


def max_pool_max_indices(embeddings, num_vec_per_polyline):
    '''
    Inputs:
    ------------------
    embeddings: [B, D, E], 
    num_vec_per_polyline: [B], the number of vectors in each polyline

    Outputs:
    ------------------
    max pooled embeddings: [B, E], the max pooled embeddings
    '''
    B, D, E = embeddings.shape
    mask = torch.arange(D).view(1, 1, -1).repeat(B, 1, 1).to(embeddings.device) < num_vec_per_polyline.view(-1, 1, 1)
    return max_pool_mask(embeddings, mask)


def max_pool_mask(embeddings, mask):
    '''
    Inputs:
    ------------------
    embeddings: [B, D, E], 
    mask: [B, R, D], 0s and 1s, 1 indicates membership

    Outputs:
    ------------------
    max pooled embeddings: [B, R, E], the max pooled embeddings according to the membership in mask
    '''
    B, D, E = embeddings.shape
    _, R, _ = mask.shape
    # extend embedding with placeholder
    embeddings_ = torch.cat([-1e6*torch.ones_like(embeddings[:, :1, :]), embeddings], dim=1)
    # transform mask to index
    index = torch.arange(1, D+1).view(1, 1, -1).repeat(B, R, 1).to(mask.device) * mask# [B, R, D]
    # batch indices
    batch_indices = torch.arange(B).view(B, 1, 1).repeat(1, R, D).to(mask.device)
    # retrieve embeddings by index
    indexed = embeddings_[batch_indices.flatten(), index.flatten(), :].view(B, R, D, E)# [B, R, D, E]
    # return
    max_pooled = indexed.max(dim=-2).values if D > 0 else indexed.view(B, R, E)
    return max_pooled


@POSITIONAL_ENCODING.register_module()
class SineContinuousPositionalEncoding(BaseModule):
    def __init__(self, 
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale
    
    def forward(self, x):
        """
        x: [B, N, D]

        return: [B, N, D * num_feats]
        """
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x


@FEEDFORWARD_NETWORK.register_module()
class SubgraphNet_Layer(BaseModule):
    def __init__(self, 
                 input_channels=256, 
                 hidden_channels=128,
                 init_cfg=None,):
        super(SubgraphNet_Layer, self).__init__(init_cfg)
        self.fc = nn.Linear(input_channels, hidden_channels)
        self.norm_layer = nn.LayerNorm(hidden_channels)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, input, num_vec_per_polyline):
        """
        input: (num_polylines, max_num_vectors, 5)
        num_vec_per_polyline: (num_polylines)
        """
        max_num_vectors = input.shape[1]
        embedd_data = self.fc(input)   # (num_polylines, max_num_vectors, hidden_channels)
        embedd_data = F.relu(self.norm_layer(embedd_data))
        polyline_feature = max_pool_max_indices(embedd_data, num_vec_per_polyline)  # (num_polylines, 1, hidden_channels)
        # kernel_size = embedd_data.shape[0]  # num_polylines
        # polyline_feature = F.max_pool1d(embedd_data.transpose(0,1), kernel_size=kernel_size)  # (hidden_channels, 1)
        polyline_feature = polyline_feature.repeat(1, max_num_vectors, 1)  # (num_polylines, max_num_vectors, hidden_channels)
        output = torch.cat([embedd_data, polyline_feature], dim=-1)   # (num_polylines, max_num_vectors, hidden_channels*2)
        return output


@BACKBONES.register_module()
class SubgraphNet(BaseModule):
    def __init__(self, 
                 input_channels,
                 num_ffns=3,
                 ffn_cfgs=dict(
                     type='SubgraphNet_Layer',
                     input_channels=256,
                     hidden_channels=128,
                 ),
                 init_cfg=None,):
        super(SubgraphNet, self).__init__(init_cfg)
        self.input_channels = input_channels
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ConfigDict(ffn_cfgs)) for _ in range(num_ffns)]

        self.ffns = ModuleList()
        for ffn_index in range(num_ffns):
            if ffn_index == 0:
                ffn_cfgs[0]['input_channels'] = self.input_channels

            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index]))
        # self.sublayer1 = SubgraphNet_Layer(input_channels)
        # self.sublayer2 = SubgraphNet_Layer()
        # self.sublayer3 = SubgraphNet_Layer() #output = 128

    def forward(self, input, num_vec_per_polyline):
        """
        input: (num_polylines, max_num_vectors, 5)
        num_vec_per_polyline: (num_polylines)
        return: (output_dim 128)
        """
        x = input
        for ffn in self.ffns:
            x = ffn(x, num_vec_per_polyline)       # (num_polylines, max_num_vectors, 128)
        # x = self.sublayer0(input, num_vec_per_polyline)  # (num_polylines, max_num_vectors, 128)
        # x = self.sublayer1(x, num_vec_per_polyline)      # (num_polylines, max_num_vectors, 128)
        # x = self.sublayer2(x, num_vec_per_polyline)      # (num_polylines, max_num_vectors, 128)
        polyline_feature = max_pool_max_indices(x, num_vec_per_polyline).squeeze()  # (num_polylines, 128)
        # kernel_size = x.shape[1]   # max_num_vectors
        # polyline_feature = F.max_pool1d(x.transpose(1,2), kernel_size=kernel_size).squeeze()  # polyline_feature.shape -> (num_polylines, 128)
        return polyline_feature
    

@BACKBONES.register_module()
class GlobalGraph(BaseModule):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, 
                 input_channels, 
                 global_graph_width,
                 init_cfg=None,):
        super(GlobalGraph, self).__init__(init_cfg)
        self.in_channels = input_channels
        self.graph_width = global_graph_width
        self.q_lin = nn.Linear(input_channels, global_graph_width)
        self.k_lin = nn.Linear(input_channels, global_graph_width)
        self.v_lin = nn.Linear(input_channels, global_graph_width)

    def forward(self, x):
        # x: (num_polylines, in_feature)
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.mm(query, key.transpose(0, 1)) / np.sqrt(self.graph_width)
        attention_weights = F.softmax(scores, dim=-1)
        x = torch.mm(attention_weights, value)
        return x


@NECKS.register_module()
class VectorNet(BaseModule):
    # TODO: make parallized (use torch_geometric) 
    def __init__(self, 
                 subgraph_backbone=dict(
                    type='SubgraphNet',
                    input_channels=3,
                 ),
                 global_graph_backbone=dict(
                    type='GlobalGraph',
                    input_channels=256, 
                    global_graph_width=256,
                 ),
                 **kwargs):
        super(VectorNet, self).__init__(**kwargs)
        self.subgraph = build_backbone(subgraph_backbone)
        self.global_graph = build_backbone(global_graph_backbone)
        self.out_feature_dim = global_graph_backbone['global_graph_width']
        # self.subgraph = SubgraphNet(sub_in_feature, **kwargs)
        # self.global_graph = GlobalGraph(global_in_feature, global_graph_width, **kwargs)

    def forward(self, map_graph, map_num_poly_pnts):
        # batch_map_graph: list of polylines
        # batch, num_polylines * max_num_vectors * 3 (x, y, lane_type, other tags (?))
        # map_num_poly_pnts: list of counts
        # batch, num_polylines * 1, number of vectors in each polyline 

        batch_graph_feats = []
        for graph_polylines, num_per_polyline in zip(map_graph, map_num_poly_pnts):
            # graph_polylines:  num_polylines * max_num_vectors * 3
            # if graph_polylines.shape[0] == 0:
            #     batch_graph_feats.append(torch.empty(0, self.out_feature_dim).to(graph_polylines.device))
            # else:
            subgraph_feats = self.subgraph(graph_polylines, num_per_polyline)  # num_polylines * feat_dim
            global_graph_feats = self.global_graph(subgraph_feats)             # num_polylines * feat_dim
            batch_graph_feats.append(global_graph_feats)
        
        return batch_graph_feats


@NECKS.register_module()
class MapGraphTransformer(BaseModule):
    def __init__(self, 
                 input_dim=30,  # 2 * 11 points + 8 classes
                 dmodel=256,
                 hidden_dim=2048,  # set this to something smaller
                 nheads=8,
                 nlayers=6,
                 batch_first=False,  # set to True
                 pos_encoder=None,
                 **kwargs):
        super(MapGraphTransformer, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.map_embedding = nn.Linear(input_dim, dmodel)

        if pos_encoder is not None:
            self.use_positional_encoding = True
            self.pos_encoder = build_positional_encoding(pos_encoder)
        else:
            self.use_positional_encoding = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, 
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, map_graph, onehot_category):
        # batch_map_graph: list of polylines
        # batch, num_polylines * points * 2 (x, y)
        # onehot_category: batch, num_polylines * num_categories, onehot encoding of categories
        # TODO: make batched
        batch_graph_feats = []
        for graph_polylines, onehot_cat in zip(map_graph, onehot_category):
            if self.use_positional_encoding:
                graph_polylines = self.pos_encoder(graph_polylines)
            
            npolylines, npoints, pdim = graph_polylines.shape
            if onehot_cat.shape[1] == 0:
                graph_feat = graph_polylines.view(npolylines, npoints * pdim)
            else:
                graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim), onehot_cat], dim=-1)

            # embed features
            graph_feat = self.map_embedding(graph_feat)  # num_polylines, dmodel

            # transformer encoder
            graph_feat = self.transformer_encoder(graph_feat.unsqueeze(self.batch_dim))  # 1, num_polylines, hidden_dim
            # graph_feat = graph_feat.squeeze(self.batch_dim)  # num_polylines, hidden_dim
            batch_graph_feats.append(graph_feat.squeeze(self.batch_dim))
        return batch_graph_feats
