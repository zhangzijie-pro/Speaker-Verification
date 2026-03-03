import torch
import torch.nn as nn
from aamsoftmax import AAMSoftmax
from diarization import DiarizationPITLoss

class MultiTaskLoss(nn.Module):
    def __init__(self, embedding_dim=192, num_classes=1000, lambda_ver=1.0, lambda_diar=0.5):
        super().__init__()
        self.ver_loss = AAMSoftmax(embedding_dim, num_classes)
        self.diar_loss = DiarizationPITLoss(max_spk=10)
        self.lambda_ver = lambda_ver
        self.lambda_diar = lambda_diar

    def forward(self, emb, pred_ids, pred_activity, pred_count,
                label, target_ids, target_activity, target_count):
        ver_loss = self.ver_loss(emb, label)                   
        diar_loss = self.diar_loss(pred_ids, pred_activity, pred_count,
                                   target_ids, target_activity, target_count)
        return self.lambda_ver * ver_loss + self.lambda_diar * diar_loss