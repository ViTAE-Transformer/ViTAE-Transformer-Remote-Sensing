_base_ = [
    '../_base_/models/upernet_our_r50.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained = '../pretrain_model/resnet50-19c8e357.pth',
    backbone=dict(),
    decode_head=dict(num_classes=16,ignore_index=255), 
    auxiliary_head=dict(num_classes=16,ignore_index=255)
    )
