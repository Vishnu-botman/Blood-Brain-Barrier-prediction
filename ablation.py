import argparse
import csv
from train import run

MODELS = ["gcn", "gat", "sage"]
DEPTHS = [2, 3, 4]


def run_ablation(datasets=["bbbp", "bace"], epochs=150):
    all_rows = []
    
    for dataset in datasets:
        print(f"\n{'='*25} RUNNING {dataset.upper()} ABLATION {'='*25}")
        dataset_rows = []
        for model in MODELS:
            for depth in DEPTHS:
                print(f"--> Training {model.upper()} with L={depth} on {dataset.upper()}...")
                val, test = run(model, dataset_name=dataset, n_layers=depth, epochs=epochs, verbose=False)
                row = {"dataset": dataset.upper(), "model": model, "layers": depth, "val_auc": val, "test_auc": test}
                dataset_rows.append(row)
                all_rows.append(row)

        print(f"\n=== {dataset.upper()} RESULTS: Test ROC-AUC (Val AUC) ===")
        header = f"{'Model':<8} " + "  ".join(f"L={d:<15}" for d in DEPTHS)
        print(header)
        print("-" * len(header))
        for model in MODELS:
            cells = []
            for r in dataset_rows:
                if r["model"] == model:
                    cells.append(f"{r['test_auc']:.3f} ({r['val_auc']:.3f})")
            print(f"{model:<8} " + "  ".join(f"{c:<17}" for c in cells))

    # Save to CSV
    csv_file = "results_dual_ablation.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "layers", "val_auc", "test_auc"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAll ablation results saved to {csv_file}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bbbp", "bace", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    ds_list = ["bbbp", "bace"] if args.dataset == "both" else [args.dataset]
    run_ablation(datasets=ds_list, epochs=args.epochs)
