import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from configs.predictor_config import Config
from utils.load_data import load_embeddings, load_sequences
from datasets.contra_dataset import ContrastiveDataset
from models.contra_model import ContrastiveModel
from models.affinity_predictor import CrossAttentionModel


def main():
    cfg = Config()
    antigen_embeddings = load_embeddings(cfg.antigen_embedding_path)
    antibody_embeddings = load_embeddings(cfg.antibody_embedding_path)
    sequences = load_sequences(cfg.sabdab_pair_file_path)

    dataset = ContrastiveDataset(sequences, antibody_embeddings, antigen_embeddings)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []
    pearson_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}")
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)

        contrastive_model = ContrastiveModel(
            cfg.antibody_embedding_dim,
            cfg.antigen_embedding_dim,
            cfg.projected_embedding_dim,
        )
        contrastive_model.load_state_dict(torch.load(cfg.contrastive_model_path))

        model = CrossAttentionModel(contrastive_model, cfg.cross_attention_emb_dim)

        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        )
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), f"checkpoints/cross_attention_fold{fold+1}.pth")

        model.eval()
        predictions = []
        true_values = []
        predicted_values = []

        current_idx = 0
        with torch.no_grad():
            for batch in val_loader:
                ab_embed, ag_embed, delta_g = batch
                ab_embed = ab_embed.to(cfg.device)
                ag_embed = ag_embed.to(cfg.device)
                delta_g = delta_g.to(cfg.device)
                model = model.to(cfg.device)

                outputs = model(ab_embed, ag_embed).squeeze()
                predicted_values.extend(outputs.cpu().numpy())
                true_values.extend(delta_g.cpu().numpy())

                batch_size = delta_g.size(0)
                for i in range(batch_size):
                    global_idx = val_idx[current_idx + i]
                    predictions.append(
                        {
                            "heavy": sequences["heavy"][global_idx],
                            "light": sequences["light"][global_idx],
                            "antigen": sequences["antigen"][global_idx],
                            "delta_g": sequences["delta_g"][global_idx],
                            "predicted_output": outputs[i].item(),
                        }
                    )
                current_idx += batch_size

        predictions_df = pd.DataFrame(predictions)
        predictions_csv_path = f"predictions_fold{fold+1}.csv"
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions for Fold {fold+1} saved at {predictions_csv_path}")

        rmse = root_mean_squared_error(true_values, predicted_values, squared=False)
        pearson_corr, _ = pearsonr(true_values, predicted_values)

        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        print(f"Fold {fold+1} Pearson Correlation: {pearson_corr:.4f}")

        rmse_scores.append(rmse)
        pearson_scores.append(pearson_corr)

    avg_rmse = np.mean(rmse_scores)
    avg_pearson = np.mean(pearson_scores)

    print(f"\nAverage RMSE across {kf.n_splits} folds: {avg_rmse:.4f}")
    print(f"Average Pearson Correlation across {kf.n_splits} folds: {avg_pearson:.4f}")


if __name__ == "__main__":
    main()
