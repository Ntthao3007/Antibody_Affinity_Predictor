import torch
import pickle
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from datasets.antibody_dataset import AntibodyDataset


class AntibodyEncoder(torch.nn.Module):
    def __init__(self, pretrained_model_name="Exscientia/IgBert"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=False
        )
        self.model = BertModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )

    @torch.no_grad()
    def forward(self, sequences_heavy, sequences_light):
        paired_sequences = [
            " ".join(h) + " [SEP] " + " ".join(l)
            for h, l in zip(sequences_heavy, sequences_light)
        ]

        tokens = self.tokenizer.batch_encode_plus(
            paired_sequences,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        outputs = self.model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        residue_embeddings = outputs.last_hidden_state

        return residue_embeddings, tokens["special_tokens_mask"]


if __name__ == "__main__":
    CSV_PATH = "datasets/sabdab-pair/pairs_sabdab_converted.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AntibodyDataset(csv_path=CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    encoder = AntibodyEncoder(pretrained_model_name="Exscientia/IgBert")

    embeddings = {}
    for batch_idx, batch in enumerate(dataloader):
        sequences_heavy, sequences_light = batch
        residue_embeddings, special_tokens_mask = encoder(
            sequences_heavy, sequences_light
        )

        residue_embeddings[special_tokens_mask == 1] = 0
        sequence_embeddings_sum = residue_embeddings.sum(dim=1)

        sequence_lengths = torch.sum(special_tokens_mask == 0, dim=1)
        sequence_embeddings_batch = (
            sequence_embeddings_sum / sequence_lengths.unsqueeze(1)
        )

        start_idx = batch_idx * dataloader.batch_size
        for i in range(sequence_embeddings_batch.size(0)):
            idx = start_idx + i
            if idx >= len(dataset):
                break
            h = sequences_heavy[i]
            l = sequences_light[i]
            embeddings[(h, l)] = sequence_embeddings_batch[i].cpu()

    with open("datasets/antibody-antigen-pkl/antibody.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Processed embeddings: {len(embeddings)}")
