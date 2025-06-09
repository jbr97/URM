import torch
from torch import nn
from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

class PromptEncoder(nn.Module):
    def __init__(self, args, num_refine_steps: int, emb_dim: int, num_heads: int, layer_norm_eps: float,
                 mlp_factor: int, norm_first: bool, activation: nn.Module, norm: bool, dropout: float,):
        super(PromptEncoder, self).__init__()
        self.pos_emb = PositionalEncodingsFixed(emb_dim)
        self.prompt_pos_emb = nn.Parameter(
            torch.randn(args.num_objects * args.kernel_dim ** 2, 1, emb_dim))  # [27, 1, 256]

        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps, mlp_factor, norm_first, activation
            ) for _ in range(num_refine_steps)
        ])
        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, f_e, prompt):  # [b, 256, 64, 64] [27, b, 256]
        bs, c, h, w = f_e.size()
        pos_emb = self.pos_emb(bs, h, w, f_e.device).flatten(2).permute(2, 0, 1)  # [4096, b, 256]
        prompt_pos_embed = self.prompt_pos_emb  # [27, 1, 256]

        output = f_e.flatten(2).permute(2, 0, 1)  # [4096, b, 256]
        for i, layer in enumerate(self.layers):
            output = layer(output, prompt, pos_emb, prompt_pos_embed)
        return self.norm(output)

class IterativeAdaptationLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(IterativeAdaptationLayer, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def forward(self, tgt, prompt, pos_emb, prompt_pos_embed):
        if self.norm_first:
            tgt_norm = self.norm1(tgt)
            tgt = tgt + self.dropout1(self.enc_dec_attn(
                query=tgt_norm + pos_emb,
                key=prompt + prompt_pos_embed,
                value=prompt,
            )[0])
            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.mlp(tgt_norm))
        else:
            tgt = self.norm1(tgt + self.dropout1(self.enc_dec_attn(
                query=tgt + pos_emb,
                key=prompt + prompt_pos_embed,
                value=prompt,
            )[0]))
            tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))
        return tgt
