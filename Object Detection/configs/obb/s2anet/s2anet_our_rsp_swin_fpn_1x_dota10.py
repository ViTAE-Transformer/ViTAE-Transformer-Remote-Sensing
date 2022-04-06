_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])

#optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# RetinaNet nms is slow in early stage, disable every epoch evaluation
evaluation = None

model = dict(
    type='S2ANet',
    pretrained='/public/data3/users/wangdi153/RS_CV/Swin-Transformer-main/output/swin_tiny_patch4_window7_224/epoch120/swin_tiny_patch4_window7_224/default/ckpt.pth',
    backbone=dict(
        type='swin',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.3,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        frozen_stages=2,
        norm_eval=False
        ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='S2AHead',
        feat_channels=256,
        align_type='AlignConv',
        heads=[
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                anchor_generator=dict(
                    type='Theta0AnchorGenerator',
                    scales=[4],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            ),
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                with_orconv=True,
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            )
        ]
    )
)

# training and testing settings
train_cfg = [
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
]
test_cfg = dict(
    skip_cls=[True, False],
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8
)

# Final Result
# mAP: 0.7404564819386658
# ap of each class: plane:0.8878974204378595, baseball-diamond:0.8071806291320748, bridge:0.5260062844100715,
# ground-track-field:0.733263720865298, small-vehicle:0.7863866711912353, large-vehicle:0.7897973201340522,
# ship:0.8766842782424621, tennis-court:0.9090225563909776, basketball-court:0.8498791032236941,
# storage-tank:0.8440616190813502, soccer-ball-field:0.6071881997157468, roundabout:0.661790586216465,
# harbor:0.6948527168745157, swimming-pool:0.6631525938702733, helicopter:0.4696835292939079
