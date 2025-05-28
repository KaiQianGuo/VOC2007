# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
data_root = '/data/disk2/guokaiqian/mmdetection/data/'
# 增加类别定义
classes = (
    'aeroplane','bicycle','bird','boat','bottle','bus','car',
    'cat','chair','cow','diningtable','dog','horse',
    'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4, #2-》4
    num_workers=6, #2-》6
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        ann_file=data_root + 'coco/voc07_train.json',
        data_prefix=dict(img=data_root + 'VOCdevkit/'),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='val2017/'),
        ann_file=data_root + 'coco/voc07_val.json',
        data_prefix=dict(img=data_root + 'VOCdevkit/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        # data_prefix=dict(img='test2017/'),
        ann_file=data_root + 'coco/voc07_test.json',
        data_prefix=dict(img=data_root + 'VOCdevkit/'),
        metainfo=dict(classes=classes),
        test_mode=True,
         indices=4,  # 新增随机采样设置
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'coco/voc07_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
    outfile_prefix='./work_dirs/coco_instance/val')
# test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.

test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    # ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    ann_file=data_root + 'coco/voc07_test.json',
    outfile_prefix='./work_dirs/coco_instance/test')
