import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

class MDCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.shared_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
    def forward(self, hsi, lidar):
        diff_hsi_lidar = hsi - lidar
        diff_lidar_hsi = lidar - hsi
        weight_hsi = self.sigmoid(self.gap(diff_hsi_lidar))
        weight_lidar = self.sigmoid(self.gap(diff_lidar_hsi))
        hsi_enhanced = hsi + self.beta * weight_lidar * lidar
        lidar_enhanced = lidar + self.beta * weight_hsi * hsi
        shared = torch.cat([hsi_enhanced, lidar_enhanced], dim=1)
        shared = self.shared_proj(shared)
        return hsi_enhanced, lidar_enhanced, shared

class SVACM(nn.Module):
    def __init__(self, dim_head, heads, dim_in=64, dim_embed=128, dim_out=64):
        super().__init__()

        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.convH = nn.Conv2d(dim_in, dim_embed, 3, stride=1, padding=1)
        self.BN_H1 = nn.BatchNorm2d(dim_embed)
        self.convL = nn.Conv2d(dim_in, dim_embed, 3, stride=1, padding=1)
        self.BN_L1 = nn.BatchNorm2d(dim_embed)
        # shared embedding
        self.convS = nn.Conv2d(dim_in, dim_embed, 3, stride=1, padding=1)
        self.BN_S1 = nn.BatchNorm2d(dim_embed)
        # Q,K from H and L
        self.Hto_q = nn.Linear(dim_embed, inner_dim, bias=False)
        self.Hto_k = nn.Linear(dim_embed, inner_dim, bias=False)
        self.Lto_q = nn.Linear(dim_embed, inner_dim, bias=False)
        self.Lto_k = nn.Linear(dim_embed, inner_dim, bias=False)
        # shared -> V
        self.Sto_v = nn.Linear(dim_embed, inner_dim, bias=False)
        self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
        # proj back
        self.projH = nn.Linear(inner_dim, dim_embed, bias=True)
        self.projL = nn.Linear(inner_dim, dim_embed, bias=True)
        self.LN_H2 = nn.LayerNorm(dim_embed)
        self.LN_L2 = nn.LayerNorm(dim_embed)
        self.project_out = nn.Conv2d(dim_embed * 2, dim_out, kernel_size=1)

    def forward(self, F_H, F_L, shared):

        # feature embedding
        F_H = F.relu(self.BN_H1(self.convH(F_H)))  # [B,dim_embed,H-2,W-2]
        F_L = F.relu(self.BN_L1(self.convL(F_L)))
        S   = F.relu(self.BN_S1(self.convS(shared)))
        b, c, h, w = F_H.shape
        n = h * w
        # flatten tokens
        F_H_t = F_H.permute(0, 2, 3, 1).reshape(b, n, c)  # [B,N,dim_embed]
        F_L_t = F_L.permute(0, 2, 3, 1).reshape(b, n, c)
        S_t   = S.permute(0, 2, 3, 1).reshape(b, n, c)
        # Q,K
        Hq = rearrange(self.Hto_q(F_H_t), 'b n (h d) -> b h n d', h=self.num_heads)
        Hk = rearrange(self.Hto_k(F_H_t), 'b n (h d) -> b h n d', h=self.num_heads)
        Lq = rearrange(self.Lto_q(F_L_t), 'b n (h d) -> b h n d', h=self.num_heads)
        Lk = rearrange(self.Lto_k(F_L_t), 'b n (h d) -> b h n d', h=self.num_heads)
        Sv = rearrange(self.Sto_v(S_t), 'b n (h d) -> b h n d', h=self.num_heads)
        # normalize
        Hq = F.normalize(Hq, dim=-1)
        Hk = F.normalize(Hk, dim=-1)
        Lq = F.normalize(Lq, dim=-1)
        Lk = F.normalize(Lk, dim=-1)
        # attention (token-level)
        attnH = (Hq @ Lk.transpose(-2, -1)) * self.rescaleH  # [B,heads,N,N]
        attnH = attnH.softmax(dim=-1)
        attnL = (Lq @ Hk.transpose(-2, -1)) * self.rescaleL
        attnL = attnL.softmax(dim=-1)
        # out uses shared value
        out_H = attnH @ Sv  # [B,heads,N,d]
        out_L = attnL @ Sv
        # merge heads -> [B,N,inner_dim]
        out_H = rearrange(out_H, 'b h n d -> b n (h d)')
        out_L = rearrange(out_L, 'b h n d -> b n (h d)')
        out_H = self.projH(out_H)  # [B,N,dim_embed]
        out_L = self.projL(out_L)
        F_H_t = F_H_t + out_H
        F_L_t = F_L_t + out_L
        F_H_map = self.LN_H2(F_H_t).reshape(b, h, w, c).permute(0, 3, 1, 2)
        F_L_map = self.LN_L2(F_L_t).reshape(b, h, w, c).permute(0, 3, 1, 2)
        fusion = torch.cat([F_H_map, F_L_map], dim=1)
        fusion = self.project_out(fusion)
        return fusion

class CrossFusionBlock(nn.Module):
    def __init__(self, dim_in=64, dim_embed=128, dim_out=64, heads=8):
        super().__init__()
        assert dim_embed % heads == 0
        dim_head = dim_embed // heads

        self.cross = SVACM(
            dim_head=dim_head,
            heads=heads,
            dim_in=dim_in,
            dim_embed=dim_embed,
            dim_out=dim_out
        )

    def forward(self, hsi, lidar, shared):
        return self.cross(hsi, lidar, shared)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):  # x: (B,C,H,W)
        avg_map = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))  # (B,1,H,W)
        return attn

class SEAM(nn.Module):
    def __init__(self, channels=64, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.spatial_gate = SpatialAttention(kernel_size=7)
        self.post_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
    def forward(self, x):
        B, C, H, W = x.shape
        # ---- Sobel(Ei) ----
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        edge_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        gate = self.spatial_gate(x)
        fused = edge * gate
        out = self.post_block(fused)
        if self.use_residual:
            out = out + x
        return out

class SFEM(nn.Module):
    def __init__(self, tau=0.5, alpha=3.0, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(0.3))

        self.out_proj = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
    def calculate_spectral_distance(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
        x_norm = x_centered / (x_centered.std(dim=-1, keepdim=True) + self.eps)
        corr = torch.matmul(x_norm, x_norm.transpose(1, 2)) / (H * W)
        return 1.0 - corr
    def build_attention_matrix(self, D):
        M = torch.sigmoid((self.tau - D) * self.alpha)
        M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
        return M
    def enhance_in_frequency_domain(self, x, M):
        freq = torch.fft.fft2(x, dim=(-2, -1))
        real = torch.einsum('bij,bjhw->bihw', M, freq.real)
        imag = torch.einsum('bij,bjhw->bihw', M, freq.imag)
        freq = torch.complex(real, imag)
        return torch.fft.ifft2(freq, dim=(-2, -1)).real
    def forward(self, x):
        D = self.calculate_spectral_distance(x)
        M = self.build_attention_matrix(D)
        x_enhanced = self.enhance_in_frequency_domain(x, M)
        out = x + self.gamma * x_enhanced
        return self.out_proj(out)

class Proposed(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()

        if dataset_name == 'YRD-DVD':
            hsi_dim, lidar_dim, num_classes = 125, 1, 7
        elif dataset_name == 'YRD-MVD':
            hsi_dim, lidar_dim, num_classes = 125, 1, 6
        elif dataset_name == 'THR-MVD':
            hsi_dim, lidar_dim, num_classes = 126, 1, 5
        else:
            raise ValueError("Unknown dataset")

        self.conv3d_hsi = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )
        self.conv2d_hsi = nn.Sequential(
            nn.Conv2d(8 * hsi_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # LiDAR extractor
        self.conv2d_lidar = nn.Sequential(
            nn.Conv2d(lidar_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv2d_lidar2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.mdcm = MDCM(64)
        self.lgt_hsi = SFEM()
        self.lgt_lidar = SEAM()
        self.cross_fusion = CrossFusionBlock(dim_in=64, dim_embed=128, dim_out=64, heads=8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x1, x2):
        # HSI
        x1 = x1.unsqueeze(1)
        x1 = self.conv3d_hsi(x1)
        x1 = rearrange(x1, 'b c h w y -> b (c h) w y')
        x1 = self.conv2d_hsi(x1)
        x2 = self.conv2d_lidar(x2)
        x2 = self.conv2d_lidar2(x2)
        x1, x2, shared = self.mdcm(x1, x2)
        x1 = self.lgt_hsi(x1)
        x2 = self.lgt_lidar(x2)
        # cross fusion
        fusion = self.cross_fusion(x1, x2, shared)
        f = self.avg_pool(fusion).flatten(1)
        return self.fc(f)
