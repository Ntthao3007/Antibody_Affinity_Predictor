import torch
import pickle

from huggingface_hub import login
from torch.utils.data import DataLoader

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig

from datasets.antigen_dataset import AntigenDataset
from configs.key_config import ACCESS_TOKEN_READ


class AntigenEncoder(torch.nn.Module):
    def __init__(
        self,
        access_token: str,
        pretrained_model_name: str = "esm3-open",
    ):
        super().__init__()
        login(token=access_token)
        self.pretrained_model_name = pretrained_model_name
        self.model: ESM3InferenceClient = ESM3.from_pretrained(
            self.pretrained_model_name
        )

    @torch.no_grad()
    def forward(self, ag_seq: str):
        protein = ESMProtein(sequence=ag_seq)
        protein_tensor = self.model.encode(protein)

        output = self.model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )

        emb = output.per_residue_embedding
        del protein, protein_tensor
        return emb


if __name__ == "__main__":
    CSV_PATH = "datasets/sabdab-pair/pairs_sabdab_converted.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AntigenDataset(csv_path=CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = AntigenEncoder(access_token=ACCESS_TOKEN_READ).to(device)

    embeddings = {}
    for batch in dataloader:
        for ag_seq in batch:
            res_emb = model(ag_seq)
            seq_emb = res_emb.mean(axis=0)
            embeddings[ag_seq] = seq_emb.cpu()

    with open("datasets/antibody-antigen-pkl/antigen.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Processed embeddings: {len(embeddings)}")
