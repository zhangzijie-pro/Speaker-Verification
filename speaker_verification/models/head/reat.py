import torch
import torch.nn as nn
import torch.nn.functional as F


class REAT_DiarizationHead(nn.Module):
    """
    在线 diarization head
    输出:
      frame_embeds:    [B, T, D]
      activity_logits: [B, T]
      count_logits:    [B, K]   # 可选，说话人数分类辅助头
    """

    def __init__(self, in_dim=512, emb_dim=192, num_speakers_max=10):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.max_spk = num_speakers_max

        # 帧级 speaker embedding 投影头
        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )

        # 活动检测头（logits）
        self.activity_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

        # 说话人数辅助分类头（基于时序均值池化）
        self.count_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_speakers_max),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, in_dim]
        """
        frame_embeds = self.frame_proj(frame_feat)                 # [B,T,D]
        frame_embeds = F.normalize(frame_embeds, dim=-1)

        activity_logits = self.activity_head(frame_feat).squeeze(-1)  # [B,T]

        utt_feat = frame_feat.mean(dim=1)                         # [B,in_dim]
        count_logits = self.count_head(utt_feat)                  # [B,K]

        return frame_embeds, activity_logits, count_logits