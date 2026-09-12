"""Run the exact saved hackathon models for a new molecule; never retrain in the demo."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from rdkit import DataStructs
from torch_geometric.loader import DataLoader
from chem_data import canonical, fingerprint, graph, rd_fp
from models import MODELS


class ScreeningSystem:
    def __init__(self, artifact_dir='artifacts'):
        base = Path(artifact_dir)
        if not (base / 'summary.json').exists():
            raise FileNotFoundError(f'{base}/summary.json missing. Run python run_hackathon.py first.')
        cfg = json.loads((base / 'summary.json').read_text(encoding='utf-8'))['configuration']
        if cfg['seed'] != 42 or cfg['layers'] != 2 or cfg['dataset'] != 'bbbp':
            raise ValueError('These are not the fixed-seed 2-layer hackathon artifacts')
        folder = base / 'seed_42'
        self.models = {}
        for name in ('gcn', 'gat', 'sage'):
            model = MODELS[name](cfg['in_dim'], cfg['hidden'], 1, cfg['layers'])
            model.load_state_dict(torch.load(folder / f'{name}.pt', map_location='cpu', weights_only=True))
            self.models[name] = model.eval()
        self.rf = joblib.load(folder / 'rf.joblib')
        self.reference = joblib.load(base / 'training_fingerprints.joblib')

    @torch.no_grad()
    def predict(self, smiles):
        smi = canonical(smiles)
        if smi is None:
            raise ValueError('Enter one valid molecule as SMILES.')
        fp = fingerprint(smi).reshape(1, -1)
        batch = next(iter(DataLoader([graph(smi)], batch_size=1)))
        probs = {'rf': float(self.rf.predict_proba(fp)[0, 1])}
        for name, model in self.models.items():
            logits = model(batch.x.float(), batch.edge_index, batch.batch).view(-1)
            probs[name] = float(torch.sigmoid(logits)[0])
        members = [probs[n] for n in ('rf', 'gcn', 'gat', 'sage')]
        mean = float(np.mean(members))
        similarity = max(DataStructs.BulkTanimotoSimilarity(rd_fp(smi), self.reference), default=0.)
        votes = sum(x >= .5 for x in members)
        agree = max(votes, 4-votes)
        accepted = similarity >= .35 and agree >= 3 and (mean <= .35 or mean >= .65)
        return {'dataset': 'bbbp', 'smiles': smi,
                'decision': ('BBB+' if mean >= .5 else 'BBB-') if accepted else 'UNCERTAIN',
                'suggested_class': 'BBB+' if mean >= .5 else 'BBB-',
                'positive_class_probability': mean, 'model_positive_probabilities': probs,
                'agreement': f'{agree}/4', 'nearest_training_tanimoto': similarity,
                'in_domain': similarity >= .35, 'accepted': accepted,
                'notice': 'Uncalibrated structural-screening estimate; not an experimental or clinical result.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('smiles')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    print(json.dumps(ScreeningSystem(args.artifacts).predict(args.smiles), indent=2))
