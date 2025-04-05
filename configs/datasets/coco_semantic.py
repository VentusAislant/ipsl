# dataset settings
_base_ = "mmdet::_base_/datasets/coco_semantic.py"
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
dataset_type = {{_base_.dataset_type}}
data_root = '{{$COCO_DATASET:data/COCO/}}'
backend_args = {{_base_.backend_args}}

train_dataset = dict(
    _scope_="mmdet",
    type=dataset_type,
    data_prefix=dict(
        img_path="images/train2017/",
        seg_map_path="annotations/stuff_annotations_trainval2017/stuff_train2017_pixelmaps/",
    ),
    data_root=data_root,
    pipeline=train_pipeline
)

val_dataset = dict(
    _scope_="mmdet",
    type=dataset_type,
    data_prefix=dict(
        img_path="images/val2017/",
        seg_map_path="annotations/stuff_annotations_trainval2017/stuff_val2017_pixelmaps/",
    ),
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

val_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
