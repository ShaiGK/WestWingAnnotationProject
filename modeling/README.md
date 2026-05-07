# Modeling Power Dynamics

This folder contains the modeling pipeline for predicting power dynamics in
short two-character dialogue excerpts from *The West Wing*. The goal is to move
from our human annotation project to models that can predict the same annotation
fields from the dialogue text.

For each excerpt, we predict three kinds of labels:

- `power_rating`: a five-way label from `-2` to `+2`
- `power_shift`: a binary label for whether dominance changes during the excerpt
- `power_strategies`: a multi-label set of conversational strategies used to assert power

The `power_rating` label is interpreted from Character A's perspective:

- `+2`: Character A is very dominant
- `+1`: Character A is slightly dominant
- `0`: neither character is clearly dominant
- `-1`: Character B is slightly dominant
- `-2`: Character B is very dominant

Because the three outputs have different structures, I treat this as a
multi-task prediction problem, but I evaluate each task separately. Rating is
multiclass classification, shift is binary classification, and strategies are
multi-label classification.

## Files

- `data.py`: joins the annotations to the dialogue text and creates a
  model-ready dataset.
- `train.py`: trains the TF-IDF/logistic regression baseline.
- `predict.py`: loads the baseline model and predicts labels for one dialogue file.
- `deberta_train.py`: trains a shared DeBERTa encoder with separate task heads.
- `colab_deberta_training.ipynb`: Colab notebook for running DeBERTa on a GPU.
- `requirements.txt`: Python dependencies for the modeling code.

Generated files are ignored by Git:

- `modeling/dataset.json`
- `modeling/artifacts/`
- Python cache files

## Dataset Construction

The annotations live in `annotations/all_annotations.json`, but the actual
dialogue text lives in the `dialogues/` folder. The first step is therefore to
join each annotation to its dialogue excerpt.

The annotation IDs use this format:

```text
S01E01_BILLY-SAM_pair1_exc1
```

The corresponding dialogue filename uses this format:

```text
dialogues/season_1/S01E01_BILLY-SAM_01_01.txt
```

`data.py` handles this mapping. It also has a fallback for small naming
differences in character names, such as punctuation being removed in filenames.

If multiple annotators labeled the same excerpt, `data.py` collapses those
annotations into one training target. For `power_rating`, it uses majority vote.
For `power_shift` and each strategy label, it uses a majority threshold. Most
items currently only have one annotation, so most training labels come directly
from a single annotator.

To build the dataset:

```bash
python3 modeling/data.py
```

This creates:

```text
modeling/dataset.json
```

Each example contains metadata, the dialogue text, the final rating label, the
shift label, and the strategy labels.

## Baseline Model

The baseline model is intentionally simple and interpretable. It uses TF-IDF
features over the dialogue text, with Character A and Character B included in the
input string.

The baseline trains three separate classifiers:

- logistic regression for `power_rating`
- logistic regression for `power_shift`
- one-vs-rest logistic regression for `power_strategies`

This gives us a useful point of comparison before using a transformer model. The
dataset is small and imbalanced, so this baseline is also helpful for seeing
which labels are especially hard to learn.

To train the baseline:

```bash
python3 modeling/train.py --rebuild-dataset
```

Main outputs:

```text
modeling/artifacts/metrics.json
modeling/artifacts/baseline_predictions.csv
modeling/artifacts/power_models.joblib
```

The prediction CSV stores one row per excerpt. It includes the model name, train
or test split, true labels, predicted labels, probability scores, and the input
text. I use this file to compare baseline predictions against DeBERTa
predictions later.

To run the baseline on a single dialogue file:

```bash
python3 modeling/predict.py \
  dialogues/season_1/S01E01_BILLY-SAM_01_01.txt \
  --character-a BILLY \
  --character-b SAM
```

## DeBERTa Multi-Task Model

The stronger model is implemented in `deberta_train.py`. It uses a shared
DeBERTa encoder and three prediction heads:

- a five-class `power_rating` head
- a two-class `power_shift` head
- an eight-label `power_strategies` head

The default encoder is:

```text
microsoft/deberta-v3-base
```

The model uses weighted losses because the labels are imbalanced:

- `power_rating`: weighted cross-entropy
- `power_shift`: weighted cross-entropy
- `power_strategies`: binary cross-entropy with per-strategy positive weights

The total loss is a weighted sum of the three task losses. The default task
weights are:

```text
rating:   1.0
shift:    0.7
strategy: 1.0
```

However, our best performance with DeBERTa came from the following weights:

```text
rating:   1.0
shift:    0.3
strategy: 0.5
```

To train DeBERTa locally or on a GPU machine:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --run-name deberta_v3_base_multitask \
  --output-dir modeling/artifacts/deberta_v3_base_multitask \
  --max-length 384 \
  --batch-size 8 \
  --epochs 5 \
  --lr 1e-5
```

Main outputs:

```text
modeling/artifacts/deberta_v3_base_multitask/best_model.pt
modeling/artifacts/deberta_v3_base_multitask/latest_model.pt
modeling/artifacts/deberta_v3_base_multitask/last_checkpoint.pt
modeling/artifacts/deberta_v3_base_multitask/metrics.json
modeling/artifacts/deberta_v3_base_multitask/predictions.csv
```

`best_model.pt` is the best model by the script's selection score, which averages
rating macro F1, shift macro F1, and strategy micro F1. `latest_model.pt` is the
final epoch's model. `last_checkpoint.pt` stores the model, optimizer,
scheduler, epoch number, and metric history so training can be resumed exactly.

## Resuming DeBERTa Training

If `last_checkpoint.pt` exists, resume from it:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --run-name deberta_v3_base_multitask \
  --output-dir modeling/artifacts/deberta_v3_base_multitask \
  --max-length 384 \
  --batch-size 8 \
  --epochs 12 \
  --lr 1e-5 \
  --resume-from-checkpoint modeling/artifacts/deberta_v3_base_multitask/last_checkpoint.pt
```

Here `--epochs 12` means "train through epoch 12 total." If the checkpoint is
from epoch 5, the script will run epochs 6 through 12.

If only `best_model.pt` exists, the model can still be initialized from those
weights, but the optimizer and scheduler will start fresh:

```bash
python3 modeling/deberta_train.py \
  --dataset modeling/dataset.json \
  --model-name microsoft/deberta-v3-base \
  --run-name deberta_v3_base_multitask_continue \
  --output-dir modeling/artifacts/deberta_v3_base_multitask_continue \
  --max-length 384 \
  --batch-size 8 \
  --epochs 7 \
  --lr 5e-6 \
  --init-from-model modeling/artifacts/deberta_v3_base_multitask/best_model.pt
```

## Colab Training

The DeBERTa model is intended to be trained on a GPU. I included
`colab_deberta_training.ipynb` so the model can be trained in Google Colab using
files stored in Google Drive.

Minimal Google Drive layout:

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

Only these files are needed if `dataset.json` has already been created locally.

If I want to rebuild the dataset inside Colab, I also need:

```text
annotations/all_annotations.json
dialogues/season_*/*.txt
```

The notebook installs dependencies, mounts Drive, optionally reruns the baseline,
trains DeBERTa, and reads the resulting prediction CSVs.

## Result Files

For final comparison, I keep baseline and DeBERTa outputs under
`modeling/artifacts/`, usually with explicit names:

```text
baseline_metrics.json
baseline_predictions.csv
deberta_metrics.json
deberta_predictions.csv
```

The metrics files summarize aggregate performance. The prediction CSVs are more
useful for error analysis because they show exactly which examples each model
gets right or wrong.

I do not commit the result artifacts or model checkpoints to GitHub because they
are generated outputs and can be large. The code, notebook, and documentation are
the parts that should be version controlled.

## Notes on Evaluation

The current split is a stratified random train/test split based on
`power_rating`. I report macro F1 for rating and shift because those labels are
imbalanced. For strategies, I report both micro F1 and macro F1 because strategy
labels are sparse and multi-label.

Important limitations:

- The dataset is small for neural modeling.
- Most excerpts have only one annotation.
- The neutral class is much more common than the extreme power classes.
- Some strategy labels are rare, which makes strategy prediction difficult.

Because of those limitations, I treat the baseline and DeBERTa results as
evidence about the difficulty of the task.
