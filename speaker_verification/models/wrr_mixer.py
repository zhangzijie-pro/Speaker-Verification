import torch
import torch.nn as nn
import torch.nn.functional as F
from speaker_verification.models.wtconv import WTConv1d


import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.sqrt2_inv = 1.0 / (2 ** 0.5)

    def _haar_step(self, x):
        orig_len = x.shape[-1]
        
        if orig_len % 2 == 1:
            x = F.pad(x, (0, 1), mode='reflect')

        even = x[:, :, 0::2]
        odd  = x[:, :, 1::2]

        low  = (even + odd) * self.sqrt2_inv
        high = (even - odd) * self.sqrt2_inv

        return low, high

    def forward(self, x):
        """
        输入: x.shape = [batch, channels, time]
        """
        subbands = []
        current = x

        for _ in range(self.levels):
            low, high = self._haar_step(current)
            subbands.append(high)
            current = low

        subbands.append(current)
        return subbands[::-1]        # [最高频, ..., 最低频]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.pe[:, :x.size(1)]
        return x.transpose(1, 2)

class EPA_WRR_Mixer(nn.Module):
    def __init__(self, channels=512, wt_levels=3, heads=4):
        super().__init__()
        self.dwt = DWT(wt_levels)
        
        self.energy_fc   = nn.Linear(1, 1)
        self.energy_gate = nn.Sigmoid()
        
        self.phase_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.phase_proj = nn.Linear(channels, channels)
        
        # self.pos_encoding = PositionalEncoding(d_model=channels)
        self.scale_routers = nn.ModuleList([
            nn.Linear(channels, 1) for _ in range(wt_levels + 1)
        ])
        self.ripple_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                      dilation=2**i, groups=heads)
            for i in range(wt_levels + 1)
        ])
        self.resonance_matrix = nn.Parameter(
            torch.randn(wt_levels + 1, wt_levels + 1) / (wt_levels + 1)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.echo_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        subbands = self.dwt(x)
    
        energy = (x ** 2).mean(dim=1)                     # (B, T)
        energy_score = self.energy_fc(energy.unsqueeze(-1))  # (B, T, 1)
        gate = self.energy_gate(energy_score)             # (B, T, 1)
        gate = gate.transpose(1, 2)                       # (B, 1, T)
        x = x * gate
        
        # x = self.pos_encoding(x)
        
        phase_proxy = torch.tanh(self.phase_conv(x))
        phase_feat = self.phase_proj(phase_proxy.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        
        processed = []
        max_len = max(sb.shape[2] for sb in subbands)

        for i, sb in enumerate(subbands):
            route = torch.sigmoid(self.scale_routers[i](sb.transpose(1,2))).transpose(1,2)
            
            target_len = sb.shape[2]
            phase_resampled = F.interpolate(
                phase_feat, size=target_len, mode='linear', align_corners=False
            )
            
            sb_processed = sb * route + phase_resampled * 0.3
            sb_processed = self.ripple_convs[i](sb_processed)
            
            if sb_processed.shape[2] < max_len:
                pad = max_len - sb_processed.shape[2]
                sb_processed = F.pad(sb_processed, (0, pad), mode='constant', value=0)
            
            processed.append(sb_processed)

        fused = torch.stack(processed, dim=0)
        resonance = torch.softmax(self.resonance_matrix, dim=-1)
        fused = torch.einsum('lbct,lk->kbct', fused, resonance).mean(dim=0)
        
        global_echo = self.global_pool(fused).squeeze(-1)
        echo = self.echo_gate(global_echo).unsqueeze(-1)
        fused = fused + fused * echo
        
        out = self.proj_out(fused)
        if out.shape[2] < T:
            pad_right = T - out.shape[2]
            out = F.pad(out, (0, pad_right), mode='constant', value=0)

        elif out.shape[2] > T:
            out = out[..., :T]

        return out + x
    
class HybridEPA_WRR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, wt_levels=3):
        super().__init__()
        self.wtconv = WTConv1d(
            in_channels, out_channels,
            kernel_size=5, wt_levels=wt_levels
        )
        self.epa_wrr = EPA_WRR_Mixer(out_channels, wt_levels=wt_levels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.wtconv(x)
        x = self.epa_wrr(x)
        
        se_weight = self.se(x.mean(-1, keepdim=True))   # (B, C, 1)
        x = x * se_weight
        
        return x