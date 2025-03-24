import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ContrastiveDataset(Dataset):
    def __init__(self, sequences, antibody_embeddings, antigen_embeddings):
        self.antibody_embeddings = antibody_embeddings
        self.antigen_embeddings = antigen_embeddings
        self.pairs = [
            (((h, l), ag), delta_g)
            for h, l, ag, delta_g in tqdm(
                zip(
                    sequences["heavy"],
                    sequences["light"],
                    sequences["antigen"],
                    sequences["delta_g"],
                ),
                desc="Loading dataset",
            )
            if (h, l) in antibody_embeddings and ag in antigen_embeddings
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ((h, l), ag), delta_g = self.pairs[idx]
        ab_embed = self.antibody_embeddings[(h, l)].clone().detach()
        ag_embed = self.antigen_embeddings[ag].clone().detach()
        return ab_embed, ag_embed, float(delta_g)
