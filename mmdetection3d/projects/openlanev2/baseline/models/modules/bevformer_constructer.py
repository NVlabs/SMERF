import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils.builder import TRANSFORMER
from mmdet3d.models import NECKS

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention


@NECKS.register_module()
class BEVFormerConstructer(BaseModule):
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
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 bev_h=200,
                 bev_w=200,
                 rotate_center=[100, 100],
                 encoder=None,
                 positional_encoding=None,
                 use_map_embeds=False,
                 **kwargs):
        super(BEVFormerConstructer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_map_embeds = use_map_embeds
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rotate_center = rotate_center

        self.init_layers()

    def init_layers(self):
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        if self.use_map_embeds:
            self.map_level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
        
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
 
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        if self.use_map_embeds:
            normal_(self.map_level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # @auto_fp16(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, map_bev_feats=None, map_graph_feats=None, **kwargs):
        """
        obtain bev features.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in img_metas])
        delta_y = np.array([each['can_bus'][1]
                           for each in img_metas])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in img_metas])

        grid_length_y = self.real_h / self.bev_h
        grid_length_x = self.real_w / self.bev_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in img_metas])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam x bs x  h*w x c
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        
        # Process SD map features.
        # feat_flatten = torch.cat([feat_flatten, feat_map_raster], dim=0)  # (num_cam + 1, H*W, bs, embed_dims)
        if map_bev_feats is not None:
            map_feats_flat = []
            map_bev_spatial_shapes = []
            for lvl, mfeat in enumerate(map_bev_feats):
                bs, _, c, h, w = mfeat.shape
                mspatial_shape = (h, w)
                mfeat = mfeat.flatten(3).permute(1, 0, 3, 2)  # 1 x bs x  h*w x c
                # if self.use_map_embeds:
                #     mfeat = mfeat + self.map_embeds[:, None, None, :].to(mfeat.dtype)
                if self.use_map_embeds:
                    mfeat = mfeat + self.map_level_embeds[None,
                                                    None, lvl:lvl + 1, :].to(mfeat.dtype)
                map_bev_spatial_shapes.append(mspatial_shape)
                map_feats_flat.append(mfeat)
            
            map_feats_flat = torch.cat(map_feats_flat, 2).squeeze(0)  # only a single map
            map_bev_spatial_shapes = torch.as_tensor(
                map_bev_spatial_shapes, dtype=torch.long, device=bev_pos.device)
            map_bev_level_start_index = torch.cat((map_bev_spatial_shapes.new_zeros(
                (1,)), map_bev_spatial_shapes.prod(1).cumsum(0)[:-1]))
        else:
            map_feats_flat = None
            map_bev_spatial_shapes = None
            map_bev_level_start_index = None
        
        if map_graph_feats is not None:
            if isinstance(map_graph_feats, list):
                # batch, num_polylines * feat_dim
                map_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in map_graph_feats], 
                                                device=map_graph_feats[0].device)

                max_num_polylines = map_graph_shapes.max()
                map_graph_feats = torch.stack([
                    torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], 
                                                      mfeat.shape[1])], dim=0) for mfeat in map_graph_feats
                ])  # batch, max_num_polylines, feat_dim
            else:
                raise NotImplementedError()

            # map_graph_flat = torch.cat(map_graph_feats, dim=0)  # (batch total num_polylines , feat_dim) ~= (batch * num_polylines, feat_dim)
        else:
            map_graph_shapes = None
            map_graph_feats = None

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas,
            map_bev=map_feats_flat,
            bev_spatial_shapes=map_bev_spatial_shapes,
            bev_level_start_index=map_bev_level_start_index,
            map_graph_feats=map_graph_feats,
            map_graph_shapes=map_graph_shapes,
            **kwargs
        )

        return bev_embed

