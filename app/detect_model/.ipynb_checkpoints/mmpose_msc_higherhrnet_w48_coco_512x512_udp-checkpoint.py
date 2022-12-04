# -*- coding: utf-8 -*-
# mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py

_base_ = [
#     '../../../../_base_/default_runtime.py',
#     '../../../../_base_/datasets/coco.py'
    '/workspace/mydata/mmpose/configs/_base_/default_runtime.py',
#     '/workspace/mydata/mmpose/configs/_base_/datasets/coco_cube.py'
    '/workspace/mydata/mmpose/configs/_base_/datasets/coco_msc_tray.py'
]

#
log_interval = 50
eval_interval = 50

checkpoint_config = dict(interval=eval_interval)
evaluation = dict(interval=eval_interval, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])

#
total_epochs = 3000

channel_cfg = dict(
#     dataset_joints=17,
#     dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],],
#     inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    dataset_joints=6,
    dataset_channel=[[0, 1, 2, 3, 4, 5],],
    inference_channel=[0, 1, 2, 3, 4, 5]
)

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128, 256],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# pretrained = 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'
pretrained = '/workspace/mydata/mmdetection/checkpoints/hrnet_w48-8ef0771d.pth'

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained=pretrained,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='AEHigherResolutionHead',
        in_channels=48,
        num_joints=6, #17
        tag_per_joint=True,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,
        num_deconv_filters=[48],
        num_deconv_kernels=[4],
        num_basic_blocks=4,
        cat_output=[True],
        with_ae_loss=[True, False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=6, #17
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=False,
        align_corners=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
#         flip_test=True,
        flip_test=False,  # 추론시 속도 향상됨
        use_udp=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40,
        use_udp=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
        use_udp=True,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1], use_udp=True),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ],
        use_udp=True),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

# datatype = 'MSCPilotBottomUp'
datatype = 'MSCTrayBottomUp'
# data_root = 'data/coco'
# data_root = '/workspace/mydata/AI_Camera/data'
data_root = '/workspace/data/msc_pilot2/train_keypoint'

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=16),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
#         type='BottomUpCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
#         img_prefix=f'{data_root}/train2017/',
        type = datatype,
        ann_file=f'{data_root}/msc_keypoint_coco_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
#         type='BottomUpCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
        type = datatype,
        ann_file=f'{data_root}/msc_keypoint_coco_valid.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
#         type='BottomUpCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
        type = datatype,
        ann_file=f'{data_root}/msc_keypoint_coco_valid.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

load_from = None
# load_from = '/workspace/mydata/AI_Camera/result/mmpose/bottomup/20220914_higherhrnet_w48_udp_epoch_300.pth'
# resume_from = None
# resume_from = '/workspace/mydata/AI_Camera/result/mmpose/bottomup/20220914_higherhrnet_w48_udp_epoch_300.pth'
resume_from = '/workspace/mydata/AI_Camera/result/mmpose/bottomup/20221018_realdata3_e1000.pth'

work_dir = '/workspace/mydata/AI_Camera/result/mmpose/bottomup/'

