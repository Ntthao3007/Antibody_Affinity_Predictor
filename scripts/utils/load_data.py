import pickle
import pandas as pd


def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_sequences(file_path):
    df = pd.read_csv(file_path)
    return {
        "heavy": df["Antibody sequence_heavy"].values,
        "light": df["Antibody sequence_light"].values,
        "antigen": df["Antigen sequence"].values,
        "delta_g": df["delta_g"].values,
    }
