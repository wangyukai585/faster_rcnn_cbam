# ============================================================
# 基础配置：标准 Faster R-CNN（Baseline）
# 不含任何注意力模块，作为消融实验和改进实验的对比基准。
# 数据集：PASCAL VOC 2007 trainval + VOC 2012 trainval → 测试 VOC 2007 test
# ============================================================

# 注册自定义模块（ResNetCBAM 等），让 MMDetection 能通过名字找到它们
custom_imports = dict(
    imports=['pytorch_mmdet.models'],
    allow_failed_imports=False,
)

# ============================================================
# 模型配置
# ============================================================
model = dict(
    type='FasterRCNN',
    # 数据预处理器：负责图像归一化（ImageNet均值方差）和 padding
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # RGB 均值
        std=[58.395, 57.12, 57.375],      # RGB 标准差
        bgr_to_rgb=True,
        pad_size_divisor=32,              # padding 到 32 的倍数，适配 FPN stride
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),         # 输出 layer1~layer4 的特征图
        frozen_stages=1,                  # 冻结 stem + layer1，节省显存
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,                   # BN 在训练时保持 eval 状态（常见做法）
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            # 优先用本地文件；若下载过 resnet50 可改为绝对路径
            # 下载命令：wget https://download.pytorch.org/models/resnet50-0676ba61.pth
            #           -O ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
            checkpoint='torchvision://resnet50',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # ResNet50 四个 stage 的输出通道
        out_channels=256,
        num_outs=5,                           # 输出 P2~P6，P6 由 P5 最大池化得到
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],       # 对应 FPN 的 P2~P6
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,                   # VOC 共 20 个类别
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    # 训练阶段的 RPN / RoI 采样配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    # 测试阶段的 NMS 配置
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)

# ============================================================
# 数据集配置
# ============================================================
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

# 训练流水线：Resize → RandomFlip → Pack（归一化由 data_preprocessor 完成）
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

# 测试流水线：Resize → Pack
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# 训练集：VOC2007 trainval + VOC2012 trainval 合并
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=None,
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=None,
            ),
        ],
    ),
)

# 验证集 / 测试集：VOC2007 test
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)
test_dataloader = val_dataloader

# ============================================================
# 评估器：VOC mAP（11点插值法）
# ============================================================
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

# ============================================================
# 训练策略
# ============================================================
max_epochs = 12

# EpochBasedTrainLoop：按 epoch 迭代，val_interval=1 表示每 epoch 评估一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# SGD 优化器：lr=0.01，weight_decay=1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
)

# 学习率调度：在第 8、11 epoch 各乘以 0.1
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    )
]

# ============================================================
# Hook 配置
# ============================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),         # 每 50 iter 打印一次
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                                       # 每 epoch 保存一次
        save_best='pascal_voc/mAP',                      # 同时保存最优 mAP 权重
        rule='greater',
        max_keep_ckpts=2,                                 # 最多保留 2 个 checkpoint
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# ============================================================
# 运行环境配置
# ============================================================
default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
