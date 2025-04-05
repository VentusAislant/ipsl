_base_ = [
    "../mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
]

pretrained = "pretrained_models/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.pth"

model = dict(init_cfg=dict(type='Pretrained',checkpoint=pretrained))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    _scope_="mmdet",
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

org_batch_size=2
batch_size=4
train_dataloader = dict(batch_size=org_batch_size)

# 368750 == 50 epoch
# learning policy
data_decrease_scale = 5 * (batch_size // org_batch_size)
max_iters = 368750 // data_decrease_scale
param_scheduler = dict(
    _scope_="mmdet",
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778 // data_decrease_scale, 355092 // data_decrease_scale],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 5000 // data_decrease_scale
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    _scope_="mmdet",
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        _scope_="mmdet",
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval))
log_processor = dict(_scope_="mmdet", type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
