import pandas as pd
from torch.utils.data import Dataset


class AntigenDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.antigens = df["Antigen sequence"].values

    def __len__(self):
        return len(self.antigens)

    def __getitem__(self, idx):
        antigen = self.antigens[idx]
        return antigen
