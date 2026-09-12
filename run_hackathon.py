"""One-command BBBP Track 3 experiment with no unrelated molecular labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import joblib
import numpy as np
import torch
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.loader import DataLoader
from chem_data import load_fixed_splits, generate_scaffold_splits, fingerprint, rd_fp, graph
from metrics import scores
from models import MODELS

SEED = 42
NAMES = ('gcn', 'gat', 'sage')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_loader(records, batch_size=64, shuffle=False):
    return DataLoader([graph(s, y) for s, y in records], batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_gnn(model, loader, device):
    model.eval()
    result = []
    for b in loader:
        b = b.to(device)
        logits = model(b.x.float(), b.edge_index, b.batch).view(-1)
        result.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(result)


def train_gnn(name, train, valid, args, folder, device):
    # Restore seed before every architecture so runs are repeatable.
    seed_everything()
    model = MODELS[name](graph(train[0][0]).x.shape[1], args.hidden, 1, 2, dropout=.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    train_loader = data_loader(train, args.batch, shuffle=True)
    valid_loader = data_loader(valid, args.batch)
    labels = [y for _, y in valid]
    best_score, best_state, best_epoch, waiting = -1., None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x.float(), batch.edge_index, batch.batch).view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
        val_auc = scores(labels, predict_gnn(model, valid_loader, device))['roc_auc']
        if val_auc > best_score + 1e-5:
            best_score, best_epoch, waiting = val_auc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            waiting += 1
        if waiting >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, folder / f'{name}.pt')
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name}: best validation ROC-AUC {best_score:.4f}, epoch {best_epoch}, parameters {count:,}')
    return model, count, best_epoch


def ensemble_outputs(predictions, rows, training_fps):
    matrix = np.stack([predictions[name] for name in ('rf', *NAMES)])
    mean = matrix.mean(axis=0)
    votes = (matrix >= .5).sum(axis=0)
    agree = np.maximum(votes, 4-votes)
    sims = np.array([max(DataStructs.BulkTanimotoSimilarity(rd_fp(s), training_fps), default=0.) for s, _ in rows])
    accepted = (sims >= .35) & (agree >= 3) & ((mean <= .35) | (mean >= .65))
    return mean, accepted


def record_metrics(rows, probabilities, training_fps):
    labels = np.array([y for _, y in rows])
    output = {name: scores(labels, arr) for name, arr in probabilities.items()}
    gnn_mean = np.mean([probabilities[name] for name in NAMES], axis=0)
    final, accepted = ensemble_outputs(probabilities, rows, training_fps)
    output['gnn_only_ablation'] = scores(labels, gnn_mean)
    output['proposed_ensemble'] = scores(labels, final)
    output['accepted_only'] = {'coverage': float(accepted.mean()),
                               'metrics': scores(labels[accepted], final[accepted])}
    return output


def make_report(summary):
    cfg, result = summary['configuration'], summary['results']
    m = result['test']
    ordered = ['gcn', 'gat', 'sage', 'rf', 'gnn_only_ablation', 'proposed_ensemble']
    def pretty(v):
        return f'{v:.4f}' if v is not None else 'N/A'
    table = ['| Method | Test ROC-AUC | Test PR-AUC | Balanced accuracy |',
             '|---|---:|---:|---:|']
    for name in ordered:
        x = m[name]
        table.append(f"| {name} | {pretty(x['roc_auc'])} | {pretty(x['pr_auc'])} | {pretty(x['balanced_accuracy'])} |")
    passed = m['proposed_ensemble']['roc_auc'] > m['gcn']['roc_auc']
    coverage = m['accepted_only']['coverage']
    return f'''# Track 3 BBBP Molecular Screening Report Draft

## Problem
Predict BBB permeability (BBB+ vs BBB−) from an input molecular graph. The task is graph-level binary classification. Dataset: **MoleculeNet BBBP**. Primary metric: **test ROC-AUC**, subject to organizer confirmation. Split origin: **{cfg['split_origin']}**. Fixed file hashes and counts are in `summary.json`.

## Method
Mandatory baseline: **2-layer GCN**. Comparison GNNs: 2-layer GAT (four heads) and GraphSAGE. Proposed method: mean of GCN, GAT, GraphSAGE and Morgan-fingerprint Random Forest probabilities. Ablation: mean of the three GNNs without RF. Every method uses the same fixed train/validation/test BBBP molecules. Seed 42; early stopping selects GNN checkpoints on validation ROC-AUC. RF fits training only. The test set is evaluated once after selection. The demo additionally abstains for disagreement, low similarity or probabilities close to .5; leaderboard ROC-AUC uses predictions for **every** test molecule.

## Results
{chr(10).join(table)}

Primary outcome: {'proposed method exceeded' if passed else 'proposed method did NOT exceed'} the mandatory GCN on this test split. Proposed minus GCN ROC-AUC: {m['proposed_ensemble']['roc_auc']-m['gcn']['roc_auc']:+.4f}. The ablation tests whether adding RF to GNN consensus changes performance; it does not establish a causal explanation. The demo abstention rule accepted {coverage:.1%} of test molecules; report coverage beside accepted-only metrics. Parameter counts: {cfg['parameters']}.

## Limitations and citations to complete
Structural features do not capture every biological mechanism. The decision cutoffs (.35/.65 and Tanimoto .35) are heuristics, and ensemble probabilities are not calibrated. One fixed split and seed do not establish universal performance. If this is the self-generated split, it must not be presented as organizer-provided. **Fill in:** dataset citation and organizer requirements; original repository authors/reference code; any licenses or other external assets. Do not submit this report without verifying those details and exporting to a PDF of at most four pages. Never change this table to hide a losing result.
'''


def run_dataset(args):
    if args.epochs < 1 or args.patience < 1:
        raise ValueError('epochs and patience must be positive')
    dataset = 'bbbp'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.official_data_dir:
        base = Path(args.official_data_dir)
        fixed_dir = base / dataset if (base / dataset).is_dir() else base
        split_origin = 'organizer-provided fixed splits; verify their provenance'
        preparation = None
    else:
        source = Path(__file__).parent / 'data_sources' / f'{dataset}.csv'
        fixed_dir = out / 'splits'
        preparation = generate_scaffold_splits(source, fixed_dir, seed=SEED)
        split_origin = 'locally generated seed-42 scaffold split from the uploaded raw CSV; NOT organizer-provided'
    parts, paths = load_fixed_splits(fixed_dir)
    folder = out / 'seed_42'
    folder.mkdir()
    files = {key: {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'count': len(parts[key])} for key, path in paths.items()}
    manifest = {'dataset': dataset, 'split_origin': split_origin,
                'source_preparation': preparation, 'split_files': files}
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    seed_everything()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'{dataset.upper()} ({split_origin}): train {len(parts["train"])} / valid {len(parts["valid"])} / test {len(parts["test"])}; seed=42; device={device}')
    train_x = np.stack([fingerprint(s) for s, _ in parts['train']])
    train_y = np.array([y for _, y in parts['train']])
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=SEED,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(train_x, train_y)
    joblib.dump(rf, folder / 'rf.joblib')
    training_fps = [rd_fp(s) for s, _ in parts['train']]
    joblib.dump(training_fps, out / 'training_fingerprints.joblib')
    models, parameters, epochs = {}, {}, {}
    for name in NAMES:
        models[name], parameters[name], epochs[name] = train_gnn(name, parts['train'], parts['valid'], args, folder, device)
    result = {}
    # The test DataLoader is built for the first time only after fitting and checkpoint selection.
    for split in ('valid', 'test'):
        rows = parts[split]
        loader = data_loader(rows, args.batch)
        fps = np.stack([fingerprint(s) for s, _ in rows])
        probabilities = {'rf': rf.predict_proba(fps)[:, 1]}
        for name in NAMES:
            probabilities[name] = predict_gnn(models[name], loader, device)
        result[split] = record_metrics(rows, probabilities, training_fps)
    config = {'track': 3, 'dataset': dataset, 'split_origin': split_origin,
              'seed': SEED, 'primary_metric': 'roc_auc', 'hidden': args.hidden,
              'in_dim': graph(parts['train'][0][0]).x.shape[1], 'layers': 2, 'heads': 4,
              'parameters': parameters, 'best_epochs': epochs, 'split_files': files,
              'decision_thresholds': {'probability_low': .35, 'probability_high': .65, 'min_similarity': .35, 'min_agree': 3}}
    summary = {'configuration': config, 'results': result}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'report_draft.md').write_text(make_report(summary), encoding='utf-8')
    print(f'{dataset.upper()} test ROC-AUC GCN:', result['test']['gcn']['roc_auc'])
    print(f'{dataset.upper()} test ROC-AUC proposed:', result['test']['proposed_ensemble']['roc_auc'])
    print(f'{dataset.upper()} beat mandatory GCN:', result['test']['proposed_ensemble']['roc_auc'] > result['test']['gcn']['roc_auc'])
    print(f'Saved checkpoints, metrics and report draft to {out}')


def main(args):
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} is not empty; choose a fresh --out directory for a reproducible run')
    run_dataset(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Track 3 fixed-split BBBP training and evaluation')
    p.add_argument('--official-data-dir', help='optional folder with organizer-provided BBBP train/valid/test CSVs')
    p.add_argument('--out', default='artifacts')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--cpu', action='store_true')
    main(p.parse_args())
