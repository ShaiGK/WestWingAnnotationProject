# Modeling Power Dynamics

This folder contains a first-pass multi-task baseline for predicting:

- `power_rating`: one of `-2`, `-1`, `0`, `+1`, `+2`
- `power_shift`: binary
- `power_strategies`: multi-label binary predictions, one label per strategy

The current dataset is small and imbalanced, so this starts with a transparent
baseline: TF-IDF dialogue features plus separate logistic regression classifiers
for each task. That is the right first checkpoint before trying a larger
transformer, because it gives you a reproducible score and makes label problems
easy to see.

## Recommended Approach

Treat this as multi-task learning conceptually, but evaluate each task separately.
The labels have different structures:

- Power rating is ordinal multiclass. The baseline trains it as multiclass; a
  later improvement could use ordinal regression or a regression loss rounded
  back to the five labels.
- Power shift is binary and rare, so use class balancing and report macro F1,
  not only accuracy.
- Power strategies are multi-label. Each strategy is its own binary target, and
  examples can have zero, one, or many strategies.

For multiple annotations on the same excerpt, `data.py` collapses labels by
majority vote. Since most excerpts currently have only one annotation, these
models mostly learn from single annotator labels.

## Commands

Build the model-ready dataset:

```bash
python3 modeling/data.py
```

Train and evaluate the baseline:

```bash
python3 modeling/train.py --rebuild-dataset
```

This also saves per-example predictions to:

```text
modeling/artifacts/baseline_predictions.csv
```

Predict one dialogue file:

```bash
python3 modeling/predict.py dialogues/season_1/S01E01_BILLY-SAM_01_01.txt --character-a BILLY --character-b SAM
```

Outputs:

- `modeling/dataset.json`: joined annotations plus dialogue text
- `modeling/artifacts/power_models.joblib`: trained models
- `modeling/artifacts/metrics.json`: train/test metrics
- `modeling/artifacts/baseline_predictions.csv`: comparable per-example predictions

Train the DeBERTa multi-task model:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --output-dir modeling/artifacts/deberta_v3_base_multitask \
  --batch-size 8 \
  --epochs 5 \
  --lr 1e-5
```

The DeBERTa script uses one shared encoder and three heads. Rating and shift use
weighted cross-entropy; strategies use binary cross-entropy with per-strategy
positive weights. The outputs are:

- `best_model.pt`
- `latest_model.pt`
- `last_checkpoint.pt`
- `metrics.json`
- `predictions.csv`
- tokenizer files and `config.json`

To continue a run after this checkpointing support is enabled:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --output-dir modeling/artifacts/deberta_v3_base_multitask \
  --epochs 12 \
  --resume-from-checkpoint modeling/artifacts/deberta_v3_base_multitask/last_checkpoint.pt
```

Here `--epochs 12` means train through epoch 12 total. If the checkpoint is from
epoch 5, the script will run epochs 6 through 12.

If you only have an older `best_model.pt`, you can still continue from those
weights, but the optimizer and learning-rate scheduler will start fresh:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --output-dir modeling/artifacts/deberta_v3_base_multitask_continue \
  --epochs 7 \
  --init-from-model modeling/artifacts/deberta_v3_base_multitask/best_model.pt
```

## Colab Training

Use `modeling/colab_deberta_training.ipynb` on a GPU runtime.

Minimal files to upload to Google Drive:

- `modeling/dataset.json`
- `modeling/data.py`
- `modeling/deberta_train.py`
- `modeling/train.py`
- `modeling/predict.py`
- `modeling/__init__.py`
- `modeling/requirements.txt`

Recommended Drive layout:

```text
MyDrive/
└── WestWingPower/
    └── modeling/
        ├── __init__.py
        ├── data.py
        ├── dataset.json
        ├── deberta_train.py
        ├── predict.py
        ├── requirements.txt
        └── train.py
```

If you want to rebuild `dataset.json` inside Colab, also upload:

- `annotations/all_annotations.json`
- all files under `dialogues/season_*/*.txt`

## Next Steps

Good upgrades after this baseline:

- Split by episode instead of random excerpt once the dataset grows, to test
  whether the model generalizes beyond local episode context.
- Add speaker-aware features, such as turn counts, interruptions, questions,
  imperatives, and whether A or B speaks first/last.
- Try a transformer encoder with shared text embeddings and three heads:
  multiclass rating, binary shift, and multi-label strategies.
- Use annotator disagreement as signal: keep soft labels for repeated excerpts
  instead of forcing a hard majority label.
