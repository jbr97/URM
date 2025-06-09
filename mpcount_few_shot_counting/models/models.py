import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y

def upsample(x, scale_factor=2, mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    
class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5):
        super().__init__()

        self.den_dropout = den_dropout

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout)
        )

    def forward_fe(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)

        return y_cat, x3
    
    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        d = self.den_head(y_den)
        d = upsample(d, scale_factor=4)

        return d
    
class DGModel_mem(DGModel_base):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5):
        super().__init__(pretrained, den_dropout)

        self.mem_size = mem_size
        self.mem_dim = mem_dim

        self.mem = nn.Parameter(torch.FloatTensor(1, self.mem_dim, self.mem_size).normal_(0.0, 1.0))

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, self.mem_dim, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout)
        )

        self.den_head = nn.Sequential(
            ConvBlock(self.mem_dim, 1, kernel_size=1, padding=0)
        )

    def forward_mem(self, y):
      
        b, k, h, w = y.shape
        m = self.mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.view(b, k, -1)
        logits = torch.bmm(m_key, y_) / sqrt(k)
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.view(b, k, h, w)

        return y_new_, logits
    
    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)
        d = self.den_head(y_den_new)

        d = upsample(d, scale_factor=4)

        return d
    
class DGModel_memadd(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, err_thrs=0.5):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)

        self.err_thrs = err_thrs

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=True)
        )

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2):
        y_cat1, _ = self.forward_fe(img1)
        y_cat2, _ = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()

        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)

        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)

        d1 = upsample(d1, scale_factor=4)
        d2 = upsample(d2, scale_factor=4)

        return d1, d2, loss_con
    
class DGModel_cls(DGModel_base):
    def __init__(self, pretrained=True, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')
    
    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)
    
    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)

        return dc, c
    
class DGModel_memcls(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')
    
    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)
    
    def forward(self, x, c_gt=None):

        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den_new)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)
        return dc, c
    
class DGModel_final(DGModel_memcls):
    def __init__(self, pretrained=True, 
                 mem_size=1024, 
                 mem_dim=256, 
                 cls_thrs=0.5, 
                 err_thrs=0.5, 
                 den_dropout=0.5, 
                 cls_dropout=0.3, 
                 emb_dim = 256,
                 kernel_dim=3,
                 num_objects=3,
                 has_err_loss=False,
                 zero_shot=False):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, cls_dropout, cls_thrs)

        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss
        self.zero_shot = zero_shot

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, self.mem_dim, kernel_size=1, padding=0, bn=True)
        )
        self.emb_dim = 512
        self.ope_layers = nn.ModuleList([
            OPEModule(
                1, 256, kernel_dim, num_objects, 8,
                4, 1e-5, 8, True, nn.GELU, True, self.zero_shot
            ),
            OPEModule(
                1, 512, kernel_dim, num_objects, 8,
                8, 1e-5, 8, True, nn.GELU, True, self.zero_shot
            ),
            OPEModule(
                1, 512, kernel_dim, num_objects, 8,
                16, 1e-5, 8, True, nn.GELU, True, self.zero_shot
            ),
        ])
        
        
        self.pos_emb = PositionalEncodingsFixed(self.emb_dim)
        self.pos_emb1 = PositionalEncodingsFixed(256)
        self.kernel_dim = 3
        self.fuse_layers = nn.ModuleList([
            prompt2img_layer(
                256, 8, 0.1, 1e-5, 0.5, True, nn.GELU()
            ) ,
            prompt2img_layer(
                self.emb_dim,8, 0.1, 1e-5, 0.5, True, nn.GELU()
            ) ,
            prompt2img_layer(
                self.emb_dim, 8, 0.1, 1e-5, 0.5, True, nn.GELU()
            ) 
        ])
        self.den_head = DensityMapRegressor(256,8)

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd
    

    def forward_ope(self,x,bboxes,fuse_layer,ope_layer):
        bs,c,h,w = x.size()
        pos_emb = self.pos_emb(bs, h, w, x.device).flatten(2).permute(2, 0, 1)
        if x.shape[1] == 256:
            pos_emb = self.pos_emb1(bs, h, w, x.device).flatten(2).permute(2, 0, 1)
            all_prototypes = ope_layer(x, pos_emb, bboxes)
        else:
            pos_emb = self.pos_emb(bs, h, w, x.device).flatten(2).permute(2, 0, 1)
            all_prototypes = ope_layer(x, pos_emb, bboxes)

        f_e = x.flatten(2).permute(2,0,1)
        fuse_output = fuse_layer(f_e,all_prototypes[-1])
        return fuse_output.permute(1,2,0).reshape(bs,c,h,w)



    def forward_fe(self, x,bboxes):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x3 = self.forward_ope(x3,bboxes,self.fuse_layers[2],self.ope_layers[2])
        x2 = self.forward_ope(x2,bboxes,self.fuse_layers[1],self.ope_layers[1])
        x1 = self.forward_ope(x1,bboxes,self.fuse_layers[0],self.ope_layers[0])   
        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)

        return y_cat, x3


    def forward(self, x, gt_bboxes, c_gt=None):
   
        y_cat, x3 = self.forward_fe(x,gt_bboxes)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den_new)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)

        return dc, c

    def forward_train(self, img1, img2,gt_bboxes, c_gt=None):
        y_cat1, x3_1 = self.forward_fe(img1,gt_bboxes)
        y_cat2, x3_2 = self.forward_fe(img2,gt_bboxes)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()
        loss_err = F.l1_loss(y_in1, y_in2) if self.has_err_loss else 0

        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)

        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)

        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)

        c_resized_gt = self.transform_cls_map_gt(c_gt)
        c_resized1 = self.transform_cls_map_pred(c1)
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(c_resized1 - c_resized2)
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1)

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)
        dc1 = upsample(d1 * c_resized, scale_factor=4)
        dc2 = upsample(d2 * c_resized, scale_factor=4)
        c_err = upsample(c_err, scale_factor=4)

        return dc1, dc2, c1, c2, c_err, loss_con, loss_err
    




class prompt2img_layer(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, layer_norm_eps: float,
                 mlp_factor: int, norm_first: bool, activation: nn.Module,):
        super(prompt2img_layer, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.mlp = MLP(emb_dim, int(mlp_factor * emb_dim), dropout, activation)

    def forward(self, tgt, prompt):
        if self.norm_first:
            tgt_norm = self.norm1(tgt)
            tgt = tgt + self.dropout1(self.enc_dec_attn(
                query=tgt_norm,
                key=prompt,
                value=prompt,
            )[0])
            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.mlp(tgt_norm))
        else:
            tgt = self.norm1(tgt + self.dropout1(self.enc_dec_attn(
                query=tgt,
                key=prompt,
                value=prompt,
            )[0]))
            tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))
        return tgt



class MLP(nn.Module):
    def __init__(self,input_dim,med_dim,dropout,activate= nn.GELU()): 
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim,med_dim)
        self.fc2 = nn.Linear(med_dim,input_dim)
        self.act = activate
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out



class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=True):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU() if leaky else nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DensityMapRegressor(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(DensityMapRegressor, self).__init__()
        if reduction == 8:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.LeakyReLU()
            )
        elif reduction == 16:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                UpsamplingLayer(32, 16),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.LeakyReLU()
            )

        self.reset_parameters()

    def forward(self, x):
        return self.regressor(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
