# BBBP Blood Brain Barrier Screening

A focused third-year Track 3 graph-classification hackathon project: enter one molecular SMILES and estimate whether the molecule is **BBB+** or **BBB−**. This ZIP includes **only the BBBP dataset**, extracted as `data_sources/bbbp.csv` from your uploaded `data.zip`. Original source: `data/bbbp/raw/BBBP.csv`, 2,050 rows, columns `smiles` and `p_np`; the positive label is `p_np = 1`. `data_sources/PROVENANCE.json` records source and extracted-file SHA-256 hashes. **BACE, HIV and ClinTox data are excluded and not used in training.** Cached `.pt` files from the upload are not loaded.

The experiment trains a mandatory **2-layer GCN baseline**, a 2-layer GAT, a 2-layer GraphSAGE and a Morgan-fingerprint Random Forest. The proposed method averages their four probabilities. A GNN-only average is the ablation (RF removed). The final test ROC-AUC is computed on **all** test molecules and compared honestly with GCN; the extra `UNCERTAIN` demo output never changes the leaderboard metric.

## Install and run

Use Python 3.10–3.12 and install the appropriate [PyTorch CPU/CUDA build](https://pytorch.org/get-started/locally/) first.

```bash
pip install -r requirements.txt
python run_hackathon.py
streamlit run app.py
```

`python run_hackathon.py` is the one-command experiment: it prepares a fixed split, trains the models, evaluates test once after validation-based checkpoint selection, saves weights and prints baseline versus proposed ROC-AUC. For a second experiment, use `--out second_run` instead of overwriting the first. To inspect a trained model from the command line:

```bash
python predict.py 'CCO'
python visualize_attention.py 'CCO' --out attention.png
```

## Dataset and split honesty

Your archive contains raw BBBP data but **no organizer-issued train/validation/test split**. With seed 42, the code validates molecules, removes canonical duplicates and conflicting labels, then generates *one project-defined 80/10/10 Bemis–Murcko scaffold split*. It records dropped-row counts, the split CSVs and their hashes in `artifacts/split_manifest.json`. Every model uses this same split. This is **not** an official organizer split; do not call it one.

If your teacher later supplies fixed BBBP files, copy the unchanged `train.csv`, `valid.csv`, `test.csv` to `official_data/` and run:

```bash
python run_hackathon.py --official-data-dir official_data --out official_run
```

Required columns: `SMILES,label` with values 0 or 1, or `SMILES,BBB+/BBB-` with labels `BBB+`/`BBB-`. The loader fails rather than moving rows when exact canonical structures overlap across official splits. Use the organizer's metric and data protocol if they differ from the draft.

## Files you get after training

- `artifacts/splits/` — generated train/valid/test CSVs when using the included dataset.
- `artifacts/split_manifest.json` — split origin, source preprocessing and hashes.
- `artifacts/seed_42/` — saved GCN, GAT, GraphSAGE and RF models.
- `artifacts/summary.json` — test ROC-AUC, PR-AUC, balanced accuracy, confusion matrix, parameter counts, RF ablation and abstention coverage.
- `artifacts/report_draft.md` — results table with mandatory GCN comparison and an honest win/loss statement. Complete citations and export to **four pages maximum**.

The app loads the saved BBBP models without retraining, draws the molecule and shows per-model estimates, similarity to the training set and `BBB+`, `BBB−` or `UNCERTAIN`. It abstains heuristically when fewer than 3/4 models agree, nearest training Tanimoto similarity is below .35, or the mean probability is between .35 and .65. Probabilities are **uncalibrated**; attention visualization is qualitative, not proof of chemical feature importance. Structural screening does not replace experimental BBB testing.

No trained weights, fabricated score or prerecorded demo are included because training could not execute in this environment. After running, verify the generated results, cite MoleculeNet/BBBP and the original project's model code, then record the demo using `DEMO_GUIDE.md`. Report a negative result if the proposal does not beat GCN.
