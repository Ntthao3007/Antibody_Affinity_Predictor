import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class CrossAttentionModel(pl.LightningModule):
    def __init__(self, contrastive_model, embed_dim):
        super().__init__()
        self.ab_encoder = contrastive_model.ab_encoder
        self.ag_encoder = contrastive_model.ag_encoder

        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_pearson = torchmetrics.PearsonCorrCoef()
        self.val_pearson = torchmetrics.PearsonCorrCoef()

    def forward(self, ab_embed, ag_embed):
        ab_proj = self.ab_encoder(ab_embed).unsqueeze(1)
        ag_proj = self.ag_encoder(ag_embed).unsqueeze(1)

        attn_ab, _ = self.cross_attention(ab_proj, ag_proj, ag_proj)
        attn_ag, _ = self.cross_attention(ag_proj, ab_proj, ab_proj)

        combined = attn_ab.squeeze(1) + attn_ag.squeeze(1)
        return self.mlp(combined).squeeze(-1)

    def training_step(self, batch, batch_idx):
        ab, ag, delta_g = batch
        preds = self(ab, ag)
        loss = F.mse_loss(preds, delta_g)
        self.train_rmse(preds, delta_g)
        self.train_pearson(preds, delta_g)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ab, ag, delta_g = batch
        preds = self(ab, ag)
        loss = F.mse_loss(preds, delta_g)
        self.val_rmse(preds, delta_g)
        self.val_pearson(preds, delta_g)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
