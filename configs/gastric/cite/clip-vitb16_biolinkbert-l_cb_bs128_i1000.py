_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/iter_based_runtime.py',
    '../_base_/gastric_dataset.py',
    '../_base_/sgd_i1000_lr0.001-cos.py'
]

lr = 0.03
n = 'all'
vpl = 1
run_name = f'clip-vitb16_biolinkbert-l_vpl{vpl}_p{n}-cb_bs128_i1000_lr{lr}'

arch = 'ViT-B/16'
TEXTS = ['Well differentiated tubular adenocarcinoma',
         'Moderately differentiated tubular adenocarcinoma',
         'Poorly differentiated adenocarcinoma']
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedCLIPImageBackbone',
        arch=arch,
        prompt_length=vpl,
        prompt_pos='prepend',
        prompt_init='token'),
    neck=dict(
        type='ProjectionNeck',
        in_features=512,
        out_features=1024,
        init='normal_identity',
        float16=True),
    head=dict(
        type='TextEmbeddingHead',
        texts=TEXTS,
        temperature=4.6052,
        float16=True,
        text_encoder=dict(
            type='BERT',
            model='michiyasunaga/BioLinkBERT-large')))

optimizer = dict(lr=lr)

data = dict(
    samples_per_gpu=64,  # use 2 gpus
    train=dict(
        ann_file=f'data/gastric_cls3_ann/train_{n}_0.2.txt',
        patch_balance=True))

work_dir = f'work_dirs/gastric_cite/{run_name}'
