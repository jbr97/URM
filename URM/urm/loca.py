import torch
from torch import nn
from torch.nn import functional as F

from .ope import OPEModule
from .prompt_encoder import PromptEncoder
from .universal_prompt_attention import UniversalPromptAttn
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

class LOCA(nn.Module):
    def __init__(self,
        args,
        image_size: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        backbone_type: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):
        super(LOCA, self).__init__()
        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.residual = args.residual

        self.fuse_number = args.fuse_number
        self.load_backbone(backbone_name, backbone_type, reduction, train_backbone)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, emb_dim, kernel_size=1)
        self.pos_emb = PositionalEncodingsFixed(emb_dim)

        print('prompt encoder layers:', args.num_refine_steps)
        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )
        self.prompt_encoder = PromptEncoder(
            args, args.num_refine_steps, emb_dim, num_heads,
            layer_norm_eps, mlp_factor, norm_first, activation, norm, dropout
        )

        print('tri-prompts encoder layers:', args.universal_prompts_layers)
        self.universal_prompts_layers = args.universal_prompts_layers
        self.universal_prompts_refine_layers = args.universal_prompts_refine_layers
        self.UniversalPromptAttn = UniversalPromptAttn(
            args.universal_prompts_layers, args.universal_prompts_refine_layers, emb_dim,
            num_heads, layer_norm_eps, mlp_factor, norm_first, activation, norm, dropout,
        )

        self.no_vanilla_prompt = args.no_vanilla_prompt
        self.load_universal_prompts(args, emb_dim)

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction) for _ in range(self.universal_prompts_layers - 1)
        ])

    def load_universal_prompts(self, args, emb_dim):
        self.vanilla_prompts_number = args.num_objects * args.kernel_dim ** 2
        self.vision_prompts_number, self.language_prompts_number = \
            args.vision_prompts_number, args.language_prompts_number
        if self.no_vanilla_prompt:
            print('no vanilla prompt in tri-prompts')
            self.tri_prompts_number = args.vision_prompts_number + args.language_prompts_number
        else:
            self.tri_prompts_number = self.vanilla_prompts_number + args.vision_prompts_number + args.language_prompts_number
        self.tri_prompts_pos_emb = nn.Parameter(torch.randn(self.tri_prompts_number, 1, emb_dim))
        print('vanilla prompts number:', self.vanilla_prompts_number)
        print('vision prompts number:', self.vision_prompts_number)
        print('language_prompts_number:', self.language_prompts_number)

        self.vision_prompts = nn.Parameter(torch.randn(self.vision_prompts_number, 1, emb_dim))
        self.language_prompts = nn.Parameter(torch.randn(self.language_prompts_number, 1, emb_dim))
        self.vanilla_prompt_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.kaiming_normal_(self.vanilla_prompt_proj.weight, a=0, mode='fan_out')
        self.vision_prompt_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.kaiming_normal_(self.vision_prompt_proj.weight, a=0, mode='fan_out')
        self.language_prompt_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.kaiming_normal_(self.language_prompt_proj.weight, a=0, mode='fan_out')

        self.vision_repre_projs, self.language_repre_projs = nn.ModuleList(), nn.ModuleList()
        for i in range(self.universal_prompts_layers):
            self.vision_repre_projs.append(nn.Linear(emb_dim, 512))
            self.language_repre_projs.append(nn.Linear(emb_dim, 512))
        self.universal_vision_prompts_dropout = nn.Dropout(args.universal_vision_prompts_dropout)
        self.universal_language_prompts_dropout = nn.Dropout(args.universal_language_prompts_dropout)

    def forward(self, x, bboxes):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction  # [b, 3, 512, 512] (64, 64)
        x = self.forward_stem(x)
        layer1, layer2, layer3, layer4 = self.forward_backbone_layers(x)
        src = self.forward_fpn(layer1, layer2, layer3, layer4, size)
        src = self.input_proj(src)
        bs, c, h, w = src.size()  # [b, 256, 64, 64]

        ''' get refined vanilla prompts / prompt to image attention '''  # q: prompts, k\v: f_e
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)  # [4096, b, 256]
        prototypes = self.ope(src, pos_emb, bboxes)  # [27, b, 256]

        ''' get refined f_e / Prompt Encoder '''
        src = self.prompt_encoder(src, prototypes)  # [4096, b, 256]

        ''' Vanilla prompts '''
        vanilla_prompts = self.vanilla_prompt_proj(prototypes)

        ''' Universal prompts '''
        vision_prompts = self.vision_prompts.expand(-1, bs, -1)
        language_prompts = self.language_prompts.expand(-1, bs, -1)
        vision_prompts = self.vision_prompt_proj(vision_prompts)
        language_prompts = self.language_prompt_proj(language_prompts)
        vision_prompts = self.universal_vision_prompts_dropout(vision_prompts)
        language_prompts = self.universal_language_prompts_dropout(language_prompts)

        ''' TriPrompts '''
        if self.no_vanilla_prompt:
            tri_prompts = torch.cat((vision_prompts, language_prompts), dim=0)
        else:
            tri_prompts = torch.cat((vanilla_prompts, vision_prompts, language_prompts), dim=0)  # [33, b, 256]
        tri_prompts_pos_emb = self.tri_prompts_pos_emb.expand(-1, bs, -1)

        ''' Tri-prompts Two Way Interaction '''
        src, prompts = self.UniversalPromptAttn(src, tri_prompts, pos_emb, tri_prompts_pos_emb)

        ''' Head '''
        output = src[-1]
        output = output.permute(1, 2, 0)  # [b, 256, 4096]
        output = output.reshape(bs, self.emb_dim, h, w)
        pred_dmaps = self.regression_head(output)

        ''' Aux Head '''
        aux_dmaps = list()
        for i, output in enumerate(src[: -1]):
            output = output.permute(1, 2, 0)
            output = output.reshape(bs, self.emb_dim, h, w)
            aux_dmap = self.aux_heads[i](output)
            aux_dmaps.append(aux_dmap)

        ''' Universal Representation '''
        vision_representations, language_representations = [], []
        for i, prompt in enumerate(prompts):
            if self.no_vanilla_prompt:
                vision_repre, language_repre = \
                    prompt[: self.vision_prompts_number, :, :], \
                    prompt[self.vision_prompts_number: self.tri_prompts_number, :, :]
            else:
                vision_repre, language_repre = \
                    prompt[self.vanilla_prompts_number: self.vanilla_prompts_number + self.vision_prompts_number, :, :], \
                    prompt[self.vanilla_prompts_number + self.vision_prompts_number: self.tri_prompts_number, :, :]
            vision_repre, language_repre = \
                self.vision_repre_projs[i](vision_repre), \
                self.language_repre_projs[i](language_repre)
            vision_repre, language_repre = \
                torch.mean(vision_repre, dim=0), torch.mean(language_repre, dim=0)
            vision_repre, language_repre = \
                vision_repre / vision_repre.norm(dim=-1, keepdim=True), \
                language_repre / language_repre.norm(dim=-1, keepdim=True)
            vision_representations.append(vision_repre)
            language_representations.append(language_repre)
        return pred_dmaps, aux_dmaps, vision_representations, language_representations

    def forward_fpn(self, layer1, layer2, layer3, layer4, size):
        if self.fuse_number == 3:
            src = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [layer2, layer3, layer4]
            ], dim=1)  # [b, 3584, 64, 64]
        elif self.fuse_number == 4:
            src = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [layer1, layer2, layer3, layer4]
            ], dim=1)  # [b, 3840, 64, 64]
        return src

    def forward_stem(self, x):
        x = self.backbone.backbone.conv1(x)
        x = self.backbone.backbone.bn1(x)
        x = self.backbone.backbone.relu(x)
        x = self.backbone.backbone.maxpool(x)
        return x

    def forward_backbone_layers(self, x):
        layer1 = self.backbone.backbone.layer1(x)  # [b, 256, 128, 128]
        layer2 = self.backbone.backbone.layer2(layer1)  # [b, 512, 64, 64]
        layer3 = self.backbone.backbone.layer3(layer2)  # [b, 1024, 32, 32]
        layer4 = self.backbone.backbone.layer4(layer3)  # [b, 2048, 16, 16]
        return layer1, layer2, layer3, layer4

    def load_backbone(self, backbone_name, backbone_type, reduction, train_backbone):
        if 'resnet' in backbone_name:
            from backbone.resnet import Backbone
            self.backbone = Backbone(
                backbone_name, pretrained=False, dilation=False, reduction=reduction,
                backbone_type=backbone_type, requires_grad=train_backbone, fuse_number=self.fuse_number
            )
        else:
            raise

def build_model(args):
    return LOCA(
        args,
        image_size=args.image_size,
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
    )
