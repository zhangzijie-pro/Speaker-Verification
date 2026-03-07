import torch
import torch.nn.functional as F

from speaker_verification.online_memory import SpeakerMemory


def contiguous_regions(mask: torch.Tensor):
    """
    mask: [T] bool
    return list[(s,e)]  左闭右开
    """
    regions = []
    T = mask.numel()
    i = 0
    while i < T:
        if mask[i]:
            s = i
            while i < T and mask[i]:
                i += 1
            e = i
            regions.append((s, e))
        else:
            i += 1
    return regions


class StreamingDiarizer:
    def __init__(
        self,
        model,
        device="cpu",
        activity_th=0.5,
        match_th=0.72,
        update_momentum=0.90,
        min_segment_frames=8,
        max_speakers=128,
        median_width=5,
    ):
        self.model = model
        self.device = device
        self.activity_th = float(activity_th)
        self.min_segment_frames = int(min_segment_frames)
        self.median_width = int(median_width)

        self.memory = SpeakerMemory(
            emb_dim=192,
            device=device,
            match_th=match_th,
            update_momentum=update_momentum,
            max_speakers=max_speakers,
        )

    def reset(self):
        self.memory.reset()

    def _smooth_activity(self, prob: torch.Tensor):
        """
        prob: [T]
        简单滑窗平滑，避免 activity 抖动
        """
        if self.median_width <= 1:
            return prob

        pad = self.median_width // 2
        x = prob.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=self.median_width, stride=1)
        return x.squeeze(0).squeeze(0)

    @torch.no_grad()
    def infer_chunk(self, fbank_chunk: torch.Tensor):
        """
        fbank_chunk: [T,80]
        return dict:
        {
            "activity_prob": [T],
            "segments": [
                {"start": s, "end": e, "speaker_id": id, "score": sim, "is_new": bool}
            ]
        }
        """
        self.model.eval()

        if fbank_chunk.dim() != 2:
            raise ValueError(f"fbank_chunk should be [T,80], got {tuple(fbank_chunk.shape)}")

        x = fbank_chunk.unsqueeze(0).to(self.device)  # [1,T,80]
        _, frame_embeds, activity_logits, _ = self.model(x, return_diarization=True)

        frame_embeds = frame_embeds[0]                    # [T,D]
        activity_prob = torch.sigmoid(activity_logits[0]) # [T]
        activity_prob = self._smooth_activity(activity_prob)

        active = activity_prob >= self.activity_th
        regions = contiguous_regions(active)

        results = []
        for s, e in regions:
            if e - s < self.min_segment_frames:
                continue

            seg_emb = frame_embeds[s:e].mean(dim=0)   # [D]
            seg_emb = F.normalize(seg_emb, dim=-1)

            spk_id, score, is_new = self.memory.match(seg_emb)

            results.append({
                "start": int(s),
                "end": int(e),
                "speaker_id": int(spk_id),
                "score": float(score),
                "is_new": bool(is_new),
            })

        return {
            "activity_prob": activity_prob.detach().cpu(),
            "segments": results,
        }