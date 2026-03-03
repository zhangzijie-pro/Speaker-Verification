import torch
import torch.nn as nn
import torch.nn.functional as F

class DiarizationPITLoss(nn.Module):
    def __init__(self, max_spk=10):
        self.max_spk = max_spk
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_ids, pred_activity, pred_count, target_ids, target_activity, target_count):
        B, T = pred_ids.shape
        loss_id = 0.0
        for b in range(B):
            active_mask = target_activity[B] > 0.6
            perm_loss = []
            for p in range(self.max_spk):
                shifted = torch.roll(pred_ids[b], p, dims=0)
                perm_loss.append(self.ce(shifted[active_mask], target_ids[b][active_mask]))
            loss_id += min(perm_loss)
        loss_id /= B
        
        loss_act = self.bce(pred_activity, target_activity)
        loss_cnt = self.mse(pred_count.float(), target_count.float())
        
        return loss_id + 0.5 * loss_act + 0.3 * loss_cnt