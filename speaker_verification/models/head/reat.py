import torch
import torch.nn as nn
import torch.nn.functional as F


class REAT_DiarizationHead(nn.Module):
    """
    训练阶段输出:
      frame_embeds:    [B, T, D]              # 给在线匹配/记忆库用
      frame_logits:    [B, T, C]              # 给训练和DER评估用
      activity_logits: [B, T]
      count_logits:    [B, K]                 # K = max simultaneous speakers
    """

    def __init__(self, in_dim=512, emb_dim=192, num_classes=1000, max_mix_speakers=5):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.max_mix_speakers = max_mix_speakers

        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )

        self.frame_cls = nn.Linear(emb_dim, num_classes)

        self.activity_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

        self.count_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, max_mix_speakers),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, in_dim]
        """
        frame_embeds = self.frame_proj(frame_feat)              # [B,T,D]
        frame_embeds = F.normalize(frame_embeds, dim=-1)

        frame_logits = self.frame_cls(frame_embeds)             # [B,T,C]

        activity_logits = self.activity_head(frame_feat).squeeze(-1)  # [B,T]

        utt_feat = frame_feat.mean(dim=1)                       # [B,in_dim]
        count_logits = self.count_head(utt_feat)                # [B,K]

        return frame_embeds, frame_logits, activity_logits, count_logits