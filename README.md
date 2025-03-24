# Antibody_Affinity_Predictor

## Training

### Contrastive Model

Train the contrastive learning model:

```bash
python scripts/train_contrastive.py
```

### Affinity Predictor Model

After training the contrastive model, train the affinity predictor:

```bash
python scripts/train_predictor.py
```

### Combined Training

Run full training pipeline:

```bash
./run_full_training.sh
```
