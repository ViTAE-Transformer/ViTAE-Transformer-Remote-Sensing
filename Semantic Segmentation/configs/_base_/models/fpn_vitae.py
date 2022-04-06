# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='12345678',
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
        RC_group=[1, 16, 32, 64]
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 64, 64, 64],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
