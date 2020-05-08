# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化
        style='pytorch'),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        # anchor_ratios=[0.5, 1.0, 2.0],
        anchor_ratios=[0.1,0.2,0.5,1.0,2.0,5.0,10.0], # 甘蔗数据集的框宽高比
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    # RPN得到的框经过 bbox_roi_extractor 变成一样尺度对应的特征
    # 输出结果: proposal数 * out_channels * roi_layer['out_size'] * roi_layer['out_size']   建议框数目 * 256 * 7 * 7 (本例子)
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIPool', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    # bbox_head 执行用来 回归得到框 和 框中物体分类
    bbox_head=dict(
        type='SharedFCBBoxHead',         # 全连接层类型
        num_fcs=2,                       # 全连接层数量
        in_channels=256,                 # 输入通道数     对应上边bbox_roi_extractor设置的out_channels
        fc_out_channels=1024,            # 输出通道数
        roi_feat_size=7,                 # ROI特征层尺寸  对应上边bbox_roi_extractor设置的roi_layer['out_size']
        num_classes=1+1,                 # 分类数  背景类 + 物体类
        target_means=[0., 0., 0., 0.],   # 均值
        target_stds=[0.1, 0.1, 0.2, 0.2],# 方差
        reg_class_agnostic=False,        # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',          # RPN网络的正负样本划分
            pos_iou_thr=0.7,                # 正样本的iou阈值
            neg_iou_thr=0.3,                # 负样本的iou阈值
            min_pos_iou=0.3,                # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),             # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',           # 正负样本提取器类型
            num=256,                        # 需提取的正负样本数量
            pos_fraction=0.5,               # 正样本比例
            neg_pos_ub=-1,                  # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False),     # 把ground truth加入proposal作为正样本
        allowed_border=0,
        pos_weight=-1,                      # 正样本权重，-1表示不改变原始的权重
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',  # RCNN网络正负样本划分
            pos_iou_thr=0.5,        # 正样本的iou阈值
            neg_iou_thr=0.5,        # 负样本的iou阈值
            min_pos_iou=0.5,        # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),     # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',       # 正负样本提取器类型
            num=512,                    # 需提取的正负样本数量
            pos_fraction=0.25,          # 正样本比例
            neg_pos_ub=-1,              # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=True),  # 把ground truth加入proposal作为正样本
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=True,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.1,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05) , max_per_img=100)     # max_per_img表示最终输出的det bbox数量
    # soft-nms is also supported for rcnn testing
    #       nms=dict(type='nms', iou_thr=0.5)
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)              # soft_nms参数
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 设置albu的图像增强方式和参数
# 查看文档：https://s0pypi0org.icopy.site/project/albumentations/      有像素级(不影响框)  空间级转换(影响框,所以要在上边加keymap 将操作映射到bbox上)
albu_train_transforms = [
    dict(
        type='HorizontalFlip',       # 水平翻转
        p=0.5),                      # 当前图像应用此操作的概率
    dict(
        type='VerticalFlip',         # 垂直翻转
        p=0.5),
    dict(
        type='ShiftScaleRotate',     # 旋转图片角度
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=180,
        interpolation=1,
        p=0.5),

    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',            # 随机移动输入RGB图像的每个通道的值。
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',  # 随机改变输入图像的色调、饱和度和值。
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),

    dict(type='ChannelShuffle', p=0.1),                     # 图像通道随机交换

    dict(
        type='OneOf',                                       # 联合操作,下一行定义的transforms
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),         # 使用大小为[3,blur_limit]的随机内核模糊图像
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),  # 多尺度训练 放大图片 以提升小物体检测精度
    dict(type='RandomFlip', flip_ratio=0.5),                                    # 训练时数据增强 参考mmdet/datasets/transform.py
    dict(type='Normalize', **img_norm_cfg),

    dict(type='Albu',transforms=albu_train_transforms,                          # 使用Albu进行数据增强
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.0,
             filter_lost_elements=True),
         keymap={
             'img': 'image',                # 在图像上使用
             # 'gt_masks': 'masks',
             'gt_bboxes': 'bboxes'          # 在bbox上使用
         },
        update_pad_shape=False,
        skip_img_without_anno=True
         ),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
               'pad_shape', 'scale_factor'))
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,     # 每个gpu计算的图像数量 batch_size = num_gpus * imgs_per_gpu
    workers_per_gpu=2,  # 每个gpu分配的线程数  num_workers = num_gpus * workers_per_gpu

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
# optimizer   要修改
# 优化参数，lr为学习率，momentum为动量因子，weight_decay为权重衰减因子
# 0.02为8个GPU的
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
    interval=20,             # 调整：50--->5
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/content/drive/My Drive/work_dirs/faster_rcnn_r50_xfpn_xRoIAlign_1x.py'

load_from = None
resume_from = None
workflow = [('train', 1)]