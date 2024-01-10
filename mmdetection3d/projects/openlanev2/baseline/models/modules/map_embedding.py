import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.runner.base_module import BaseModule
from mmdet3d.models import NECKS


@NECKS.register_module()
class MapEmbedSingleLayer(BaseModule):
    """Implements the BEVFormer BEV Constructer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 embed_dims,
                 map_num_types,
                 map_max_lane_num,
                 **kwargs):
        super(MapEmbedSingleLayer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.map_num_types = map_num_types
        self.map_max_lane_num = map_max_lane_num

        # self.map_type_embedding = nn.Embedding(
        #     map_num_types, self.embed_dims)
        self.map_type_embedding = nn.Parameter(torch.Tensor(
            map_num_types, self.embed_dims))
        
        # self.map_lane_num_embedding = nn.Embedding(
        #     map_max_lane_num, self.embed_dims)
        self.map_lane_num_embedding = nn.Parameter(torch.Tensor(
            map_max_lane_num, self.embed_dims))
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.map_type_embedding)
        normal_(self.map_lane_num_embedding)
        
    def forward(self, map_img):
        map_lane_types = torch.clamp(map_img[..., 0], max=self.map_num_types - 1)  # 1 x H x W  [0, ..., 7]
        map_lane_nums = torch.clamp(map_img[..., 1], max=self.map_max_lane_num - 1)  # 1 x H x W  [0, ... , 8]
        # check everything is within range

        B, mH, mW = map_img.shape[0], map_img.shape[1], map_img.shape[2]

        if map_lane_types.dtype != torch.long:
            print("**********************************")
            print("not expected dtype, mapping to -1")
            map_lane_types = torch.ones_like(map_lane_types).long() * -1
            map_lane_nums = torch.ones_like(map_lane_nums).long() * -1

        map_type_embed_feats = torch.where(
            (map_lane_types > 0).unsqueeze(-1),
            self.map_type_embedding[map_lane_types.long(), :],
            torch.zeros((B, mH, mW, self.embed_dims), device=map_lane_types.device)
        )
        map_lane_num_embed_feats = torch.where(
            (map_lane_nums > 0).unsqueeze(-1),
            self.map_lane_num_embedding[map_lane_nums.long(), :],
            torch.zeros((B, mH, mW, self.embed_dims), device=map_lane_nums.device)
        )
        map_feats = map_type_embed_feats + map_lane_num_embed_feats

        # no neck and no backbone
        return [map_feats]
    
@NECKS.register_module()
class MapEmbedMultiLayer(BaseModule):
    """Implements the BEVFormer BEV Constructer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 embed_dims,
                 map_num_types,
                 map_max_lane_num,
                 num_levels,
                 **kwargs):
        super(MapEmbedMultiLayer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.map_num_types = map_num_types
        self.map_max_lane_num = map_max_lane_num

        if self.map_num_types > 0:
            self.map_type_embedding = nn.Parameter(torch.Tensor(
                num_levels, map_num_types, int(self.embed_dims // num_levels)), requires_grad=True)
        
        if self.map_max_lane_num > 0:
            self.map_lane_num_embedding = nn.Parameter(torch.Tensor(
                num_levels, map_max_lane_num, int(self.embed_dims // num_levels)), requires_grad=True)
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.map_num_types > 0:
            normal_(self.map_type_embedding)
        if self.map_max_lane_num > 0:
            normal_(self.map_lane_num_embedding)
        
    def forward(self, map_img):
        if self.map_max_lane_num > 0:
            map_lane_types = torch.clamp(map_img[..., 0], max=self.map_num_types - 1)  # 1 x H x W  [0, ..., 7]
            map_lane_nums = torch.clamp(map_img[..., 1], max=self.map_max_lane_num - 1)  # 1 x H x W  [0, ... , 8]
        else:
            # saved as int32, so need to cast it to int64 (long)
            map_lane_types = torch.clamp(map_img, max=self.map_num_types - 1)  # .long()
            map_lane_nums = None
        # check everything is within range

        B, mH, mW = map_img.shape[0], map_img.shape[1], map_img.shape[2]

        if map_lane_types.dtype != torch.long:
            print("**********************************")
            print("not expected dtype, mapping to -1")
            map_lane_types = torch.ones_like(map_lane_types).long() * -1
            map_lane_nums = torch.ones_like(map_lane_nums).long() * -1 if map_lane_nums is not None else None


        map_type_embed_feats, map_lane_num_embed_feats = [], []
        
        for lvl in range(self.num_levels):
            # process lane type embeddings
            lvl_map_type_embedding = self.map_type_embedding[lvl, ...]

            # pull out embedding features
            lvl_type_embed_feats = torch.where(
                (map_lane_types > 0).unsqueeze(-1),
                lvl_map_type_embedding[map_lane_types, :],
                torch.zeros((B, mH, mW, int(self.embed_dims // self.num_levels)), device=map_lane_types.device)
            )

            # apply pooling according to lvl resolution
            kernel_size = 2 ** lvl
            lvl_pool_type = F.avg_pool2d(lvl_type_embed_feats.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            type_nonempty_count = F.avg_pool2d((map_lane_types > 0).unsqueeze(1).float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            
            lvl_type_embed_feats = torch.where(
                type_nonempty_count > 0, 
                lvl_pool_type / torch.clamp(type_nonempty_count, min=1.),
                torch.zeros(lvl_pool_type.shape, device=map_lane_types.device)
            )
            map_type_embed_feats.append(lvl_type_embed_feats[..., :mH, :mW])

            # process lane number embedding
            if map_lane_nums is not None:
                lvl_map_lane_num_embedding = self.map_lane_num_embedding[lvl, ...]

                lvl_lane_num_embed_feats = torch.where(
                    (map_lane_nums > 0).unsqueeze(-1),
                    lvl_map_lane_num_embedding[map_lane_nums.long(), :],
                    torch.zeros((B, mH, mW, int(self.embed_dims // self.num_levels)), device=map_lane_nums.device)
                )

                lvl_pool_num = F.avg_pool2d(lvl_lane_num_embed_feats.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                num_nonempty_count = F.avg_pool2d((map_lane_nums > 0).unsqueeze(1).float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

                lvl_lane_num_embed_feats = torch.where(
                    num_nonempty_count > 0, 
                    lvl_pool_num / torch.clamp(num_nonempty_count, min=1.),
                    torch.zeros(lvl_pool_num.shape, device=lvl_pool_num.device)
                )
                map_lane_num_embed_feats.append(lvl_lane_num_embed_feats[..., :mH, :mW])

        map_type_embed_feats = torch.cat(map_type_embed_feats, dim=1).permute(0, 2, 3, 1)  # B x mH x mW x c
        map_lane_num_embed_feats = torch.cat(map_lane_num_embed_feats, dim=1).permute(0, 2, 3, 1) if self.map_max_lane_num > 0 else 0. # B x mH x mW x c
        map_feats = map_type_embed_feats + map_lane_num_embed_feats

        # no neck and no backbone
        return [map_feats]


@NECKS.register_module()
class MapEmbedMultiLayerOneHot(MapEmbedMultiLayer):
    """Implements the BEVFormer BEV Constructer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 embed_dims,
                 map_num_types,
                 map_max_lane_num,
                 num_levels,
                 **kwargs):
        super(MapEmbedMultiLayerOneHot, self).__init__(embed_dims, 
                                                       map_num_types,
                                                       map_max_lane_num,
                                                       num_levels,
                                                       **kwargs)
        
    def forward(self, map_img):

        assert self.map_max_lane_num == 0, "Currently only support one-hot encoding for map lane types"

        # check everything is within range
        assert map_img.shape[-1] == self.map_num_types, "map_img shape should be B x mH x mW x map_num_types"
        
        map_lane_types = map_img  # B x mH x mW x map_num_types

        B, mH, mW = map_img.shape[0], map_img.shape[1], map_img.shape[2]

        map_type_embed_feats = []
        
        for lvl in range(self.num_levels):
            # process lane type embeddings
            lvl_map_type_embedding = self.map_type_embedding[lvl, ...]

            # pull out embedding features
            lvl_type_embed_feats = torch.where(
                (map_lane_types > 0).unsqueeze(-1),
                lvl_map_type_embedding[map_lane_types, :],
                torch.zeros((B, mH, mW, int(self.embed_dims // self.num_levels)), device=map_lane_types.device)
            )

            # apply pooling according to lvl resolution
            kernel_size = 2 ** lvl
            lvl_pool_type = F.avg_pool2d(lvl_type_embed_feats.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            type_nonempty_count = F.avg_pool2d((map_lane_types > 0).unsqueeze(1).float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            
            lvl_type_embed_feats = torch.where(
                type_nonempty_count > 0, 
                lvl_pool_type / torch.clamp(type_nonempty_count, min=1.),
                torch.zeros(lvl_pool_type.shape, device=map_lane_types.device)
            )
            map_type_embed_feats.append(lvl_type_embed_feats[..., :mH, :mW])


        map_type_embed_feats = torch.cat(map_type_embed_feats, dim=1).permute(0, 2, 3, 1)  # B x mH x mW x c
        map_feats = map_type_embed_feats

        # no neck and no backbone
        return [map_feats]
