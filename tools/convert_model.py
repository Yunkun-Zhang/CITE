import torch
import sys


def moco2mmcls(path):
    sd = torch.load(path)['state_dict']
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
            k = k.replace('module.base_encoder.', '')
            k = k.replace('patch_embed.proj', 'patch_embed.projection')
            k = k.replace('blocks.', 'layers.')
            k = k.replace('norm1', 'ln1')
            k = k.replace('norm2', 'ln2')
            k = k.replace('norm', 'ln1')
            k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            k = k.replace('mlp.fc2', 'ffn.layers.1')
            new_sd['backbone.' + k] = v
    torch.save(new_sd, path)


if __name__ == '__main__':
    moco2mmcls(sys.argv[1])
