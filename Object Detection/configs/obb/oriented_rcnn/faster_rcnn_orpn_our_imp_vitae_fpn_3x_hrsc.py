_base_ = [
    '../_base_/datasets/hrsc.py',
    '../_base_/schedules/schedule_3x.py',
    '../../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[24, 33])

#optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# RetinaNet nms is slow in early stage, disable every epoch evaluation
evaluation = None

model = dict(
    type='OrientedRCNN',
    pretrained='../pretrain_model/ViTAE_Window_NoShift_12_basic_stages4_14_best.pth.tar',
    backbone=dict(
        type='ViTAE_Window_NoShift_basic',
        RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
        NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
        stages=4, 
        embed_dims=[64, 64, 128, 256], 
        token_dims=[64, 128, 256, 512], 
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2], 
        NC_heads=[1, 2, 4, 8], 
        RC_heads=[1, 1, 2, 4], 
        mlp_ratio=4., 
        NC_group=[1, 32, 64, 128], 
        RC_group=[1, 16, 32, 64],
        img_size=1024, 
        window_size=7,
        drop_path_rate=0.3,
        frozen_stages=1,
        norm_eval=True
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            target_means=[.0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='OBBStandardRoIHead',
        bbox_roi_extractor=dict(
            type='OBBSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
            out_channels=256,
            extend_factor=(1.4, 1.2),
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OBBShared2FCBBoxHead',
            start_bbox_type='obb',
            end_bbox_type='obb',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='OBB2OBBDeltaXYWHTCoder',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                           loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=200,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        sampler=dict(
            type='OBBRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='obb_nms', iou_thr=0.1), max_per_img=2000))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4
)
