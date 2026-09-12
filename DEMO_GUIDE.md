# Three-minute BBBP demo

0:00–0:25 — Declare Track 3, MoleculeNet BBBP and the single task: predict BBB penetration from molecular structure.

0:25–0:55 — Show the actual `python run_hackathon.py` run and seed-42 split manifest. Say whether the split came from this project or the organizer; the uploaded data had no official split.

0:55–1:35 — Open `artifacts/report_draft.md`: read the **real** GCN versus proposed-ensemble test ROC-AUC and the GNN-only ablation. If the proposal lost, say so.

1:35–2:35 — Run `streamlit run app.py`, enter one SMILES and show BBB+/BBB− or UNCERTAIN, the four model estimates, nearest-training similarity and molecule drawing.

2:35–3:00 — Explain that abstention is heuristic, probabilities are uncalibrated and experiments are necessary before using any molecule clinically. Finish the citations and ≤4-page report before submitting the recorded demo.
