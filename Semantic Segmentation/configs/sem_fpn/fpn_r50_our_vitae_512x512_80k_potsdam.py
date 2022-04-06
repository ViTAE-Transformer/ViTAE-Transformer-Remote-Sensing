_base_ = [
    '../_base_/models/fpn_vitae.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained = '/public/data3/users/wangdi153/RS_CV/VitAE_window/output/ViTAE_Window_NoShift_12_basic_stages4_14_224/epoch40/ViTAE_Window_NoShift_12_basic_stages4_14/default/ckpt.pth',
    backbone=dict(
        img_size=512, 
        window_size=8
    ),
    decode_head=dict(num_classes=5,ignore_index=5), 
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
