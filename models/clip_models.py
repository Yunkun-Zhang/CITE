from typing import Literal, List
import torch
import torch.nn as nn
import clip
from mmcls.models.builder import BACKBONES, HEADS
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.heads import ClsHead

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@BACKBONES.register_module()
class CLIPImageBackbone(BaseBackbone):
    """CLIP image backbone."""

    def __init__(self,
                 arch: str,
                 fix: bool = True,
                 proj: bool = True,
                 float32: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        model, _ = clip.load(arch, device=DEVICE)
        if float32:
            model = model.type(torch.float32)
        self.model = model.visual
        self.dtype = model.dtype
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d) and fix:
                module.eval()
        for param in self.model.parameters():
            param.requires_grad = not fix
        if not proj:
            self.model.proj = None

    def forward(self, x):
        return self.model(x.type(self.dtype))


@BACKBONES.register_module()
class PromptedCLIPImageBackbone(CLIPImageBackbone):
    """CLIP image backbone with visual prompts."""

    def __init__(self,
                 arch: str,
                 prompt_length: int = 1,
                 prompt_layers: List[int] = None,
                 prompt_pos: Literal['prepend', 'add'] = 'prepend',
                 prompt_init: Literal['normal', 'uniform', 'zero', 'kaiming', 'token'] = 'normal',
                 **kwargs):
        super().__init__(arch, **kwargs)
        self.embed_dim = self.model.conv1.weight.shape[0]
        self.prompt_layers = [0] if prompt_layers is None else prompt_layers

        prompt = torch.empty(
            len(self.prompt_layers), prompt_length, self.embed_dim,
            dtype=self.dtype
        )
        if prompt_init == 'uniform':
            nn.init.uniform_(prompt, -0.08, 0.08)
        elif prompt_init == 'zero':
            nn.init.zeros_(prompt)
        elif prompt_init == 'kaiming':
            nn.init.kaiming_normal_(prompt)
        elif prompt_init == 'token':
            nn.init.zeros_(prompt)
            self.prompt_initialized = False
        else:
            nn.init.normal_(prompt, std=0.02)
        self.prompt = nn.Parameter(prompt, requires_grad=True)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos

    def forward(self, x):
        x = x.type(self.dtype)
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.model.class_embedding.to(self.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=self.dtype, device=x.device),
            x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # add prompts
        if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
            with torch.no_grad():
                self.prompt.data += x.mean([0, 1]).detach().clone().type(self.dtype)
            self.prompt_initialized = True
        prompt = self.prompt.unsqueeze(2).expand(-1, -1, x.shape[1], -1)

        # prompt: [layer, length, batch, embed_dim]
        if self.prompt_pos == 'prepend':
            x = torch.cat([x[:1, :, :], prompt[0, :, :, :], x[1:, :, :]], dim=0)
        for i, resblock in enumerate(self.model.transformer.resblocks):
            if i in self.prompt_layers:
                if self.prompt_pos == 'prepend':
                    x = torch.cat([
                        x[:1, :, :],
                        prompt[i, :, :, :],
                        x[1 + self.prompt_length:, :, :]
                    ], dim=0)
                elif self.prompt_pos == 'add':
                    block_length = (x.shape[0] - 1) // self.prompt_length
                    total_length = block_length * self.prompt_length
                    x = torch.cat([
                        x[:1, :, :],
                        x[1:1 + total_length, :, :] + prompt[i].unsqueeze(0).expand(
                            block_length, -1, -1, -1).reshape(-1, x.shape[1], x.shape[2]),
                        x[1 + total_length:, :, :]
                    ], dim=0)
            x = resblock(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_post(x[:, 0, :])
        if self.model.proj is not None:
            x = x @ self.model.proj

        return x


@HEADS.register_module()
class CLIPTextHead(ClsHead):
    """CLIP text head."""

    def __init__(self,
                 arch,
                 texts: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        model, _ = clip.load(arch, device=DEVICE)
        self.texts = clip.tokenize(texts).to(DEVICE)
        self.logit_scale = model.logit_scale.exp()
        with torch.no_grad():
            self.weights = model.encode_text(self.texts)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = x[-1]  # cls token
        return x

    def forward(self, x):
        x = self.pre_logits(x)
        x = x / x.norm(dim=-1, keepdim=True)
        weights = self.weights / self.weights.norm(dim=-1, keepdim=True)
        cls_score = self.logit_scale * x @ weights.t()
        return cls_score

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.forward(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, **kwargs):
        cls_score = self.forward(x)
        return super().simple_test(cls_score, **kwargs)
