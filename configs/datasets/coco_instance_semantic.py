# dataset settings
_base_ = "mmdet::_base_/datasets/coco_instance_semantic.py"
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
dataset_type = {{_base_.dataset_type}}
data_root = '{{$COCO_DATASET:data/COCO}}'
backend_args = {{_base_.backend_args}}

train_dataset = dict(
    _scope_="mmdet",
    type=dataset_type,
    ann_file="annotations/trainval2017/instances_train2017.json",
    data_prefix=dict(
        img="images/train2017",
    ),
    filter_cfg=dict(
        filter_empty_gt=True,
        min_size=32
    ),
    data_root=data_root,
    pipeline=train_pipeline
)

val_dataset = dict(
    _scope_="mmdet",
    type=dataset_type,
    ann_file="annotations/trainval2017/instances_val2017.json",
    data_prefix=dict(
        img="images/val2017",
    ),
    test_mode=True,
    data_root=data_root,
    pipeline=test_pipeline
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _scope_="mmdet",
    type='CocoMetric',
    ann_file="annotations/trainval2017/instances_val2017.json",
    metric=['bbox', 'segm'],
    format_only=False
)
test_evaluator = val_evaluator
