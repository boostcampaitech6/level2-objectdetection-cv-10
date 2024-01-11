# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# # optimizer_1
# optimizer_1 = dict(type='AdamW', lr=0.02, betas=(0.9,0.999), weight_decay=0.01)
# optimizer_1_config = dict(grad_clip=None)
# # learning policy
# lr_config_1 = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[10, 25])
# runner_1 = dict(type='EpochBasedRunner', max_epochs=12)