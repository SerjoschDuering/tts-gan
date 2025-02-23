import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# ----------------------------
# Generator and associated modules
# ----------------------------
class Generator(nn.Module):
    def __init__(self, seq_len=8760, patch_size=15, channels=1, num_classes=9,
                 latent_dim=100, cond_dim=1000, embed_dim=10, depth=3, num_heads=5,
                 forward_drop_rate=0.5, attn_drop_rate=0.5, multi_scale=True,
                 scale_small_factor=4, scale_med_factor=2, 
                 res_weight_small=0.1, res_weight_med=0.2):
        super(Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.multi_scale = multi_scale
        self.scale_small_factor = scale_small_factor
        self.scale_med_factor = scale_med_factor
        self.res_weight_small = res_weight_small
        self.res_weight_med = res_weight_med

        # Adjust the input dimension: noise vector + encoded condition
        self.input_dim = self.latent_dim + self.cond_dim
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_dim, self.seq_len * self.embed_dim)),
            nn.LeakyReLU(0.2)
        )
        
        # Positional embeddings for each scale (if enabled)
        if self.multi_scale:
            self.pos_embed_small = nn.Parameter(torch.zeros(1, self.seq_len // self.scale_small_factor, self.embed_dim))
            self.pos_embed_med = nn.Parameter(torch.zeros(1, self.seq_len // self.scale_med_factor, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        
        # Multi-scale transformer blocks for small and medium resolutions
        if self.multi_scale:
            self.blocks_small = Gen_TransformerEncoder(
                depth=self.depth,
                emb_size=self.embed_dim,
                num_heads=max(1, self.num_heads // 2),
                drop_p=self.attn_drop_rate,
                forward_drop_p=self.forward_drop_rate
            )
            self.blocks_med = Gen_TransformerEncoder(
                depth=self.depth,
                emb_size=self.embed_dim,
                num_heads=self.num_heads,
                drop_p=self.attn_drop_rate,
                forward_drop_p=self.forward_drop_rate
            )
        # The “full‐scale” branch
        self.blocks = Gen_TransformerEncoder(
            depth=self.depth,
            emb_size=self.embed_dim,
            num_heads=self.num_heads * 2,
            drop_p=self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )
        
        # Improved output processing with spectral normalization and Tanh activation.
        self.deconv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)),
            nn.Tanh()
        )
        
    def forward(self, noise, cond):
        combined = torch.cat([noise, cond], dim=1)
        x = self.l1(combined).view(-1, self.seq_len, self.embed_dim)
        if self.multi_scale:
            # Get small and medium scale representations via adaptive pooling
            x_small = F.adaptive_avg_pool1d(x.transpose(1,2), self.seq_len // self.scale_small_factor).transpose(1,2)
            x_med = F.adaptive_avg_pool1d(x.transpose(1,2), self.seq_len // self.scale_med_factor).transpose(1,2)
            
            # Add positional embeddings
            x_full = x + self.pos_embed
            x_small = x_small + self.pos_embed_small
            x_med = x_med + self.pos_embed_med
            
            # Process each branch with separate transformer encoders
            x_small = self.blocks_small(x_small)
            x_med = self.blocks_med(x_med)
            x_full = self.blocks(x_full)
            
            # Upsample small and medium to full resolution and merge via a weighted residual connection
            x_small_up = F.interpolate(x_small.transpose(1,2), size=self.seq_len, mode='linear', align_corners=False).transpose(1,2)
            x_med_up = F.interpolate(x_med.transpose(1,2), size=self.seq_len, mode='linear', align_corners=False).transpose(1,2)
            x = x_full + self.res_weight_small * x_small_up + self.res_weight_med * x_med_up
        else:
            x = x + self.pos_embed
            x = self.blocks(x)
        
        H, W = 1, self.seq_len
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0,3,1,2))
        output = output.view(-1, self.channels, H, W)
        return output

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# ----------------------------
# Discriminator and associated modules
# ----------------------------
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=100, seq_length=8760):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            nn.Linear(patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x  

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, patch_size=15, emb_size=50, seq_length=8760,
                 depth=3, n_classes=1, num_heads=5, multi_scale=True, gp_weight=10.0):
        super(Discriminator, self).__init__()
        self.gp_weight = gp_weight
        self.multi_scale = multi_scale
        
        self.patch_embedding = PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length)
        if self.multi_scale:
            # Local branch: lighter transformer
            self.transformer_local = Dis_TransformerEncoder(
                depth=max(1, depth // 2),
                emb_size=emb_size,
                drop_p=0.1,
                forward_drop_p=0.1,
                num_heads=num_heads
            )
            # Global branch: deeper transformer
            self.transformer_global = Dis_TransformerEncoder(
                depth=depth,
                emb_size=emb_size,
                drop_p=0.1,
                forward_drop_p=0.1,
                num_heads=num_heads
            )
            self.class_head_local = ClassificationHead(emb_size, n_classes)
            self.class_head_global = ClassificationHead(emb_size, n_classes)
        else:
            self.transformer = Dis_TransformerEncoder(
                depth=depth,
                emb_size=emb_size,
                drop_p=0.5,
                forward_drop_p=0.5,
                num_heads=num_heads
            )
            self.class_head = ClassificationHead(emb_size, n_classes)
    
    def forward(self, x):
        tokens = self.patch_embedding(x)
        if self.multi_scale:
            local_features = self.transformer_local(tokens)
            local_out = self.class_head_local(local_features)
            
            global_features = self.transformer_global(tokens)
            global_out = self.class_head_global(global_features)
            
            out = 0.5 * (local_out + global_out)
        else:
            features = self.transformer(tokens)
            out = self.class_head(features)
        return out

# ----------------------------
# New: Encoder module integration
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, 
                 patch_size=2, 
                 emb_size=64, 
                 seq_length=2190,
                 depth=8, 
                 num_heads=4, 
                 latent_dim=1000, 
                 drop_p=0.1, 
                 forward_drop_p=0.1):
        super(Encoder, self).__init__()
        self.patch_embed = PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length)
        self.transformer = Gen_TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            num_heads=num_heads,
            drop_p=drop_p,
            forward_drop_p=forward_drop_p
        )
        self.reduce = Reduce('b n e -> b e', reduction='mean')
        # Two linear layers for mean and log variance:
        self.fc_mu = nn.Linear(emb_size, latent_dim)
        self.fc_logvar = nn.Linear(emb_size, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        tokens = self.patch_embed(x)
        tokens = self.transformer(tokens)
        pooled = self.reduce(tokens)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar