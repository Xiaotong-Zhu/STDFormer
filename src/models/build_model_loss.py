import torch.nn as nn
from src.models.STD import STD 
from src.models.losses import STDLoss

class STD_with_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.std = STD(cfg)
        self.criterion = STDLoss(cfg)

    def forward(self, inputs, targets, is_train=True):
        outputs = self.std(inputs)
        conf_matrix, tp, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask= outputs
        total_loss, loss_task, loss_social_nce_track = self.criterion(conf_matrix, tp, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask, targets)
        return total_loss, loss_task, loss_social_nce_track, outputs

def build_model_loss(cfg):
    model = STD_with_Loss(cfg)
    return model