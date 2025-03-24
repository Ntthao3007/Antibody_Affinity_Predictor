import torch


class Config:
    antibody_embedding_path = "datasets/antibody-antigen-pkl/antibody.pkl"
    antigen_embedding_path = "datasets/antibody-antigen-pkl/antigen.pkl"
    sabdab_pair_file_path = "datasets/sabdab-pair/pairs_sabdab_converted.csv"

    contrastive_model_path = "checkpoints/ab_ag_clip.pth"

    antigen_embedding_dim = 1536
    antibody_embedding_dim = 1024
    projected_embedding_dim = 256
    cross_attention_emb_dim = projected_embedding_dim

    device = "cuda" if torch.cuda.is_available() else "cpu"
