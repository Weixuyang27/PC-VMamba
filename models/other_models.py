import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_channels=64):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._block(input_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self._block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        # Final layer
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)  # 添加 sigmoid 激活
        return out


class TransUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_channels=64, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super(TransUNet, self).__init__()

        # CNN Encoder
        self.enc1 = self._block(input_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)

        # Transformer
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.transformer = Transformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Decoder
        # self.up4 = nn.ConvTranspose2d(768, base_channels*8, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(embed_dim, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self._block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        # Final layer
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # CNN Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Transformer
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)  # Assuming 256x256 input

        # Decoder
        d4 = self.up4(x)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)
        return out


class SwinUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=256, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.):
        super(SwinUNet, self).__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Swin Transformer blocks
        self.swin_blocks = nn.ModuleList()
        for i in range(len(depths)):
            self.swin_blocks.append(SwinBlock(embed_dim * 2 ** i, depths[i], num_heads[i], window_size, mlp_ratio))

        # Decoder
        self.up4 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)

        # Final layer
        self.final = nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)

        # Swin Transformer blocks
        features = []
        for block in self.swin_blocks:
            x = block(x)
            features.append(x)

        # Decoder
        x = self.up4(features[-1])
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        return self.final(x)


class LocalMamba(nn.Module):
    def __init__(self, num_classes, input_channels=3, d_model=256, n_layers=4, dropout=0.1):
        super(LocalMamba, self).__init__()

        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, dropout) for _ in range(n_layers)
        ])

        # Final convolution
        self.final_conv = nn.Conv2d(d_model, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)

        for block in self.mamba_blocks:
            x = block(x)

        return self.final_conv(x)


class LocalVisionMamba(nn.Module):
    def __init__(self, num_classes, input_channels=3, d_model=256, n_layers=4, dropout=0.1):
        super(LocalVisionMamba, self).__init__()

        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)

        # Vision Mamba blocks
        self.vision_mamba_blocks = nn.ModuleList([
            VisionMambaBlock(d_model, dropout) for _ in range(n_layers)
        ])

        # Final convolution
        self.final_conv = nn.Conv2d(d_model, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)

        for block in self.vision_mamba_blocks:
            x = block(x)

        return self.final_conv(x)


# Helper classes for TransUNet
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]  # 传入 query, key, value
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Helper classes for SwinUNet
class SwinBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Helper classes for Mamba models
class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mamba(self.norm(x)))
        return x


class Mamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.A = nn.Parameter(torch.randn(d_model, d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        # Simplified Mamba implementation
        x = x @ self.A
        x = x @ self.B
        x = x @ self.C
        return x


class VisionMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.vision_mamba = VisionMamba(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.vision_mamba(self.norm(x)))
        return x


class VisionMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.A = nn.Parameter(torch.randn(d_model, d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_model))
        self.D = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        # Vision Mamba implementation with additional spatial attention
        x = x @ self.A
        x = x @ self.B
        x = x @ self.C
        x = x @ self.D
        return x 