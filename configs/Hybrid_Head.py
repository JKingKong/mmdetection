# model settings
model = dict(
    type='HybridDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50/10,
        num_stages=4/4,
        # out_indices=(0, 1, 2, 3),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化
        style='pytorch'),

    neck=None,
    # bbox_head 执行用来 回归得到框 和 框中物体分类
    bbox_head=dict(
        type='SharedFCBBoxHead',  # 全连接层类型
        num_fcs=2,  # 全连接层数量
        in_channels=256 * 2,  # 输入通道数     对应上边bbox_roi_extractor设置的out_channels     混合两个模型的
        fc_out_channels=1024,  # 输出通道数
        roi_feat_size=7,  # ROI特征层尺寸  对应上边bbox_roi_extractor设置的roi_layer['out_size']
        num_classes=1 + 1,  # 分类数  背景类 + 物体类
        target_means=[0., 0., 0., 0.],  # 均值
        target_stds=[0.1, 0.1, 0.2, 0.2],  # 方差
        reg_class_agnostic=False,
        # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)
# model training and testing settings
train_cfg = dict(
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False)
)

test_cfg = dict(
    rcnn=dict(
        score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05), max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), # 多尺度训练
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
    )
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# optimizer
# 学习率调整 lr=0.02为8个GPU的
optimizer = dict(type='SGD', lr=0.02/8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,             # 使用多少张图片就输出一次   100/808
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/content/drive/My Drive/work_dirs/Hybrid_Head'
load_from = None
resume_from = None
workflow = [('train', 1)]
