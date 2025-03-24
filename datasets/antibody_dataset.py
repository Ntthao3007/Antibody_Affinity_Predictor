import pandas as pd
from torch.utils.data import Dataset


class AntibodyDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.sequences_heavy = df["Antibody sequence_heavy"].values
        self.sequences_light = df["Antibody sequence_light"].values

    def __len__(self):
        return len(self.sequences_heavy)

    def __getitem__(self, idx):
        heavy_seq = self.sequences_heavy[idx]
        light_seq = self.sequences_light[idx]
        return heavy_seq, light_seq
