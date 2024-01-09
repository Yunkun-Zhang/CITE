_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/iter_based_runtime.py',
    '../_base_/gastric_dataset.py',
    '../_base_/sgd_i1000_lr0.001-cos.py'
]

run_name = f'zeroshot_clip-vitb16'

arch = 'ViT-B/16'
TEXTS = ['Well differentiated tubular adenocarcinoma',
         'Moderately differentiated tubular adenocarcinoma',
         'Poorly differentiated adenocarcinoma']
prompt = '{}'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='CLIPImageBackbone', arch=arch),
    head=dict(
        type='CLIPTextHead',
        arch=arch,
        texts=[prompt.format(t) for t in TEXTS]))

work_dir = f'work_dirs/gastric_baseline/{run_name}'
