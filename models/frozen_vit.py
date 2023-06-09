from mmcls.models import BACKBONES
from mmcls.models.backbones import VisionTransformer


@BACKBONES.register_module()
class VisionTransformerFrozen(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return super().forward(x)[0][-1]
