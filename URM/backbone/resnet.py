import torch
from torch import nn
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d

class Backbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        backbone_type: bool,
        requires_grad: bool,
        fuse_number: int = 3
    ):
        super(Backbone, self).__init__()
        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )
        pretrained_weight = torch.load('weights/resnet50-0676ba61.pth')
        resnet.load_state_dict(pretrained_weight)

        self.backbone = resnet
        self.reduction = reduction

        if name == 'resnet50':
            if backbone_type == 'swav':
                weight_path = 'weights/swav_800ep_pretrain.pth.tar'
                checkpoint = torch.load('weights/swav_800ep_pretrain.pth.tar')
            elif backbone_type == 'torch':
                weight_path = 'weights/resnet50-11ad3fa6_V2.pth'
                checkpoint = torch.load('weights/resnet50-11ad3fa6_V2.pth')
            print('backbone type:', backbone_type, weight_path)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        if fuse_number == 3:  # concatenation of layers 2, 3 and 4
            self.num_channels = 3584
        else:
            self.num_channels = 3840

        print('requires_grad in backbone:', requires_grad)
        for n, param in self.backbone.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)
