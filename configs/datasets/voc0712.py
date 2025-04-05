# dataset settings
_base_ = "mmdet::_base_/datasets/voc0712.py"
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
dataset_type = {{_base_.dataset_type}}
data_root = '{{$VOC_DATASET:data/VOCdevkit}}'
backend_args = {{_base_.backend_args}}

train_dataset_voc12 = dict(
    _scope_="mmdet",
    type=dataset_type,
    data_root=data_root,
    ann_file="VOC2012/ImageSets/Main/train.txt",
    data_prefix=dict(sub_data_root="VOC2012/"),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=train_pipeline
)

val_dataset_voc12 = dict(
    _scope_="mmdet",
    type=dataset_type,
    data_root=data_root,
    ann_file="VOC2012/ImageSets/Main/val.txt",
    data_prefix=dict(sub_data_root="VOC2012/"),
    pipeline=test_pipeline
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset_voc12,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_voc12,
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
