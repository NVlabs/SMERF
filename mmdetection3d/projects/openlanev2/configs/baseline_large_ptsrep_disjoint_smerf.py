custom_imports = dict(imports=['projects.openlanev2.baseline'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['centerline']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7


Map_size = [(-50, 50), (-25, 25)]
method_para = dict(n_points=11) # #point for each curve
sd_method_para = dict(n_points=11)
code_size = 3 * method_para['n_points']

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_num_levels_ = 4
_num_heads_ = 4
bev_h_ = 100
bev_w_ = 200

model = dict(
    type='BaselineMapGraph',
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    map_encoder=dict(
        type='MapGraphTransformer',
        input_dim=360,  # 32 * 11 + 8
        dmodel=_dim_,
        hidden_dim=_dim_,
        nheads=_num_heads_,
        nlayers=6,
        batch_first=True,
        pos_encoder=dict(
            type='SineContinuousPositionalEncoding',
            num_feats=16,  # 2 * 16 = 32 final dim
            temperature=1000,
            normalize=True,
            range=[point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]],
            offset=[point_cloud_range[0], point_cloud_range[1]],
        ),
    ),
    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_,
        num_cams=num_cams,
        embed_dims=_dim_,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        rotate_center=[bev_h_//2, bev_w_//2],
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_)),
                    dict(
                        type='MaskedCrossAttention',
                        embed_dims=_dim_,
                        num_heads=_num_heads_,),
                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn_graph', 'norm',
                                    'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),
    bbox_head=dict(
        type='TEDeformableDETRHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        test_cfg=dict(max_per_img=50)),
    pts_bbox_head=dict(
        type='LCDeformableDETRHead',
        num_classes=1,
        in_channels=_dim_,
        num_query=100,
        bev_h=bev_h_,
        bev_w=bev_w_,
        sync_cls_avg_factor=False,
        with_box_refine=False,
        with_shared_param=False,
        code_size=code_size,
        code_weights= [1.0 for i in range(code_size)],
        pc_range=point_cloud_range,
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='LaneDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0075),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    lclc_head=dict(
        type='RelationshipHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
    lcte_head=dict(
        type='RelationshipHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
    # model training and testing settings
    bbox_train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='LaneHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=1.5),
            reg_cost=dict(type='LaneL1Cost', weight=0.0075),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            ))))


train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeFrontView'),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomParameterizeLane', method='point_subsample', method_para=method_para),
    dict(type='CustomParametrizeSDMapGraph', method='even_points_onehot_type', method_para=sd_method_para),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'map_graph', 'onehot_category',
            'gt_lc', 'gt_lc_labels',
            'gt_te', 'gt_te_labels',
            'gt_topology_lclc', 'gt_topology_lcte',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus',
        ],
    )
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ResizeFrontView'),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomParametrizeSDMapGraph', method='even_points_onehot_type', method_para=sd_method_para),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'map_graph', 'onehot_category',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus',
        ],
    )
]

dataset_type = 'OpenLaneV2GraphicalSDMapDataset'
data_root = '../data/OpenLane-V2'
meta_root = '../data/OpenLane-V2'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_train_disjoint',
        # collection='data_dict_sample_train',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=train_pipeline,
        decoding_function=dict(
            type='points_prediction_decode',
            method_para=method_para,),
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val_disjoint',
        # collection='data_dict_sample_train',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=test_pipeline,
        decoding_function=dict(
            type='points_prediction_decode',
            method_para=method_para,),
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val_disjoint',
        # collection='data_dict_sample_train',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=test_pipeline,
        decoding_function=dict(
            type='points_prediction_decode',
            method_para=method_para,),
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, start=23, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]