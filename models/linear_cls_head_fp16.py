import torch
import torch.nn as nn
from mmcls.models.heads import LinearClsHead
from mmcls.models.builder import HEADS


@HEADS.register_module()
class MyLinearClsHead(LinearClsHead):
    """LinearClsHead with fp16."""

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(self.in_channels, self.num_classes, dtype=torch.float16)
