_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/iter_based_runtime.py',
    '../_base_/gastric_dataset.py',
    '../_base_/sgd_i1000_lr0.001-cos.py'
]

lr = 1e-2
n = 'all'
vpl = 1
run_name = f'vpt_clip-vitb16_vpl{vpl}_p{n}_bs128_i1000_lr{lr}'

arch = 'ViT-B/16'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedCLIPImageBackbone',
        arch=arch,
        prompt_length=vpl,
        prompt_pos='prepend'),
    head=dict(type='MyLinearClsHead', num_classes=3, in_channels=512))

optimizer = dict(lr=lr)

data = dict(train=dict(ann_file=f'data/gastric_cls3_ann/train_{n}_0.2.txt'))

work_dir = f'work_dirs/gastric_baseline/{run_name}'
