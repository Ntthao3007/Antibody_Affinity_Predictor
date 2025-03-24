import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ContrastiveModel(pl.LightningModule):
    def __init__(
        self, ab_embed_dim=1024, ag_embed_dim=1536, embed_dim=512, temperature=0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.ab_encoder = nn.Linear(ab_embed_dim, embed_dim)
        self.ag_encoder = nn.Linear(ag_embed_dim, embed_dim)

    def forward(self, ab_embed, ag_embed):
        ab_proj = F.normalize(self.ab_encoder(ab_embed), dim=-1)
        ag_proj = F.normalize(self.ag_encoder(ag_embed), dim=-1)
        return ab_proj, ag_proj

    def contrastive_loss(self, ab_proj, ag_proj):
        logits = (ab_proj @ ag_proj.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=self.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        ab_embed, ag_embed, _ = batch  # delta_g is ignored here
        ab_proj, ag_proj = self(ab_embed, ag_embed)
        loss = self.contrastive_loss(ab_proj, ag_proj)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
