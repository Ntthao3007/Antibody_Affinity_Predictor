import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from configs.contra_config import Config
from utils.load_data import load_embeddings, load_sequences
from datasets.contra_dataset import ContrastiveDataset
from models.contra_model import ContrastiveModel


def main():
    config = Config()

    antigen_embeddings = load_embeddings(config.antigen_embedding_path)
    antibody_embeddings = load_embeddings(config.antibody_embedding_path)
    sequences = load_sequences(config.sabdab_pair_file_path)

    dataset = ContrastiveDataset(sequences, antibody_embeddings, antigen_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ContrastiveModel(
        ab_embed_dim=config.antibody_embedding_dim,
        ag_embed_dim=config.antigen_embedding_dim,
        embed_dim=config.projected_embedding_dim,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
    )
    trainer.fit(model, dataloader)

    torch.save(model.state_dict(), "checkpoints/ab_ag_clip.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
