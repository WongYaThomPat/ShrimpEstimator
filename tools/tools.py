import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2, GaussianBlur
from torchvision.io import decode_image
import torch.nn.functional as F

class ShrimpDataset(Dataset):
    def __init__(self, x: list[str], y: list[np.ndarray], transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = v2.Compose([
            v2.ToImage(),                              
            v2.ToDtype(torch.float32, scale=True),    
            v2.Resize(size=(512, 512), antialias=True),
        ]) if transform is None else transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = decode_image(self.x[idx])
        heatmap = torch.from_numpy(self.y[idx]).float()
        if self.transform:
            image = self.transform(image)
        return image, heatmap


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return x * self.sigmoid(attn)


class ShrimpModelV1(nn.Module):
    def __init__(self, in_ch=3, base_channels=16):
        super().__init__()
        c1, c2, c4 = base_channels, base_channels * 2, base_channels * 4

        # --- Main Path (Semantic/Context) ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        # Bottleneck with dilation to see a bit wider without losing resolution
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

        # --- S0 Skip (The High-Res Detail Path) ---
        # Safe 4x downsampling using AvgPool to catch tiny shrimp signals
        self.s0_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_ch, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # --- S1 Skip (The Mid-Res Structural Path) ---
        self.s1_skip = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # --- Attention Modules ---
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Global Context (Squeeze-and-Excitation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(c4, c4 // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c4 // 4, c4, bias=False),
            nn.Sigmoid()
        )

        # --- Fusion & Head ---
        self.refine = nn.Sequential(
            nn.Conv2d(c4 + c1 + c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(c2, 3, kernel_size=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Main encoding path
        x1 = self.enc1(x)        # 256x256
        x2 = self.enc2(x1)       # 128x128
        bn = self.bottleneck(x2) # 128x128
        
        # Apply Global Context to the bottleneck
        gb = self.global_pool(bn).view(bn.size(0), -1)
        gb = self.global_fc(gb).view(bn.size(0), bn.size(1), 1, 1)
        bn = bn * gb
        
        # 2. Parallel Skip paths (The "Half U-Net" injections)
        s0_feat = self.s0_skip(x)   # 512 -> 128
        s1_feat = self.s1_skip(x1)  # 256 -> 128
        
        # 3. Concatenate all info at the 128x128 level
        combined = torch.cat([bn, s1_feat, s0_feat], dim=1)
        
        # 4. Refine and Apply Spatial Attention
        feat = self.refine(combined)
        feat = self.spatial_attn(feat) # Focus on WHERE the shrimps are
        
        return self.mask_head(feat)


class ShrimpModelV2(nn.Module):
    def __init__(self, in_ch=3, base_channels=16):
        super().__init__()
        c1, c2, c4 = base_channels, base_channels * 2, base_channels * 4

        # --- High-Res Entry ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # --- Spatial Attention Moved Up ---
        # We apply this to enc1 features to filter noise early
        self.early_attn = SpatialAttention(kernel_size=7)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # --- Multi-scale Dilated Bottleneck ---
        # Captures small (d=1), medium (d=2), and large (d=4) context
        self.b_d1 = nn.Conv2d(c2, c4 // 2, kernel_size=3, padding=1, dilation=1, bias=False)
        self.b_d2 = nn.Conv2d(c2, c4 // 4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.b_d4 = nn.Conv2d(c2, c4 // 4, kernel_size=3, padding=4, dilation=4, bias=False)
        self.b_norm = nn.BatchNorm2d(c4)
        self.b_relu = nn.ReLU(inplace=True)

        # --- Upgraded Skip Paths (Space-to-Depth) ---
        # Using PixelUnshuffle(4) turns [3, 512, 512] -> [3*16, 128, 128]
        self.s0_pixel_unshuffle = nn.PixelUnshuffle(4) 
        self.s0_conv = nn.Sequential(
            nn.Conv2d(in_ch * 16, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # --- Fusion & Head ---
        # Combined channels: c4 (bottleneck) + c1 (s0 skip) + c2 (enc2 skip)
        self.refine = nn.Sequential(
            nn.Conv2d(c4 + c1 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(c2, 3, kernel_size=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Early Encoding & Attention
        x1 = self.enc1(x) 
        x1 = self.early_attn(x1) # Focus early!
        
        x2 = self.enc2(x1)
        
        # 2. Multi-scale Bottleneck
        # Branching out to different dilations
        feat_d1 = self.b_d1(x2)
        feat_d2 = self.b_d2(x2)
        feat_d4 = self.b_d4(x2)
        bn = torch.cat([feat_d1, feat_d2, feat_d4], dim=1)
        bn = self.b_relu(self.b_norm(bn))
        
        # 3. High-Res Skip (No info loss)
        s0_feat = self.s0_pixel_unshuffle(x)
        s0_feat = self.s0_conv(s0_feat)
        
        # 4. Concatenate
        # Combining: Deep Semantic (bn) + Low-level Detail (s0) + Mid-level (x2)
        combined = torch.cat([bn, s0_feat, x2], dim=1)
        
        feat = self.refine(combined)
        return self.mask_head(feat)