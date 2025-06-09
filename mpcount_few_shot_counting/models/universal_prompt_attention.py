from torch import nn
from .mlp import MLP

class UniversalPromptAttn(nn.Module):
    def __init__(self, universal_prompts_layers: int, universal_prompts_refine_layers: int, emb_dim: int, num_heads: int,
                 layer_norm_eps: float, mlp_factor: int, norm_first: bool, activation: nn.Module, norm: bool,
                 dropout: float):
        super(UniversalPromptAttn, self).__init__()
        self.img2prompts_layers = nn.ModuleList([
            img2prompt_layer(
                emb_dim, num_heads, dropout, layer_norm_eps, mlp_factor, norm_first, activation
            ) for _ in range(universal_prompts_refine_layers)
        ])

        self.prompt2img_layers = nn.ModuleList([
            prompt2img_layer(
                emb_dim, num_heads, dropout, layer_norm_eps, mlp_factor, norm_first, activation
            ) for _ in range(universal_prompts_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, f_e, prompt, pos_emb, prompt_pos_emb):
        out_prompts = list()
        for img2prompt_layer in self.img2prompts_layers:
            prompt = img2prompt_layer(f_e, prompt, pos_emb, prompt_pos_emb)
        out_prompts.append(prompt)

        out_feas = list()
        output = f_e
        for prompt2img_layer in self.prompt2img_layers:
            output = prompt2img_layer(output, prompt, pos_emb, prompt_pos_emb)
            out_feas.append(output)

        return out_feas, out_prompts

class img2prompt_layer(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, layer_norm_eps: float,
                 mlp_factor: int, norm_first: bool, activation: nn.Module,):
        super(img2prompt_layer, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def forward(self, f_e, prompt, pos_emb, prompt_pos_emb):
        if self.norm_first:
            tgt_norm = self.norm1(prompt)
            tgt = prompt + self.dropout1(self.enc_dec_attn(
                query=tgt_norm + prompt_pos_emb,
                key=f_e + pos_emb,
                value=f_e,
            )[0])
            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.mlp(tgt_norm))
        else:
            tgt = self.norm1(prompt + self.dropout1(self.enc_dec_attn(
                query=prompt + prompt_pos_emb,
                key=f_e + pos_emb,
                value=f_e,
            )[0]))
            tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))
        return tgt

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
