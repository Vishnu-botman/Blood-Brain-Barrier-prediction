"""Qualitative GAT attention inspection from the exact saved model (never retrains)."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from predict import ScreeningSystem
from chem_data import graph


def plot(smiles, artifacts, output):
    system = ScreeningSystem(artifacts)
    result = system.predict(smiles)
    mol = Chem.MolFromSmiles(result['smiles'])
    item = graph(result['smiles'])
    batch = torch.zeros(item.num_nodes, dtype=torch.long)
    with torch.no_grad():
        _, attns = system.models['gat'](item.x, item.edge_index, batch, return_attention_weights=True)
    edges, weights = attns[0]
    values = np.zeros(item.num_nodes)
    for dst, weight in zip(edges[1].tolist(), weights.mean(dim=1).tolist()):
        values[dst] += weight
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(Draw.MolToImage(mol, size=(450, 300)))
    axes[0].axis('off')
    axes[0].set_title(f"GAT positive-class estimate {result['model_positive_probabilities']['gat']:.2f}")
    axes[1].bar(range(len(values)), values)
    axes[1].set_xlabel('Atom index (matches RDKit order)')
    axes[1].set_ylabel('Sum of first-layer incoming attention')
    axes[1].set_title('Qualitative attention inspection')
    fig.suptitle('Attention weights are not a causal chemical explanation')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f'Saved {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('smiles')
    p.add_argument('--artifacts', default='artifacts')
    p.add_argument('--out', default='attention.png')
    a = p.parse_args()
    plot(a.smiles, a.artifacts, a.out)
