"""
Data loading for Track 3 (Graph Neural Networks) - BBBP and BACE datasets.

BBBP = Blood-Brain Barrier Penetration (CNS drug delivery / membrane permeability)
BACE = Beta-secretase 1 inhibition (Alzheimer target binding affinity)

Both are binary graph classification benchmarks from MoleculeNet (~3,550 molecules total).
We use PyG built-in MoleculeNet loader with Bemis-Murcko scaffold splitting
(the standard, harder, and chemically realistic split for molecular data).
"""

import random
from collections import defaultdict
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SEED = 42


def get_scaffold(smiles: str) -> str:
    """Return the Murcko scaffold (core ring structure) of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=SEED):
    """
    Group molecules by scaffold, then assign whole scaffold groups to
    train/valid/test so that structurally similar molecules don't leak
    across splits. This is the standard MoleculeNet protocol.
    """
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        smiles = data.smiles
        scaffold = get_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True)

    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    n_total = len(dataset)
    n_train = int(frac_train * n_total)
    n_valid = int(frac_valid * n_total)

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, valid_idx, test_idx


def load_dataset(name="BBBP", root="./data", batch_size=32, seed=SEED):
    """
    Unified loader for MoleculeNet datasets (BBBP or BACE).
    Returns: train_loader, valid_loader, test_loader, in_dim
    """
    name_upper = name.upper()
    if name_upper not in ["BBBP", "BACE"]:
        raise ValueError(f"Unsupported dataset: {name}. Choose BBBP or BACE.")

    dataset = MoleculeNet(root=root, name=name_upper)
    train_idx, valid_idx, test_idx = scaffold_split(dataset, seed=seed)

    train_set = dataset[train_idx]
    valid_set = dataset[valid_idx]
    test_set = dataset[test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"[{name_upper}] Loaded: {len(train_set)} train / {len(valid_set)} valid / {len(test_set)} test")
    print(f"[{name_upper}] Node feature dim: {dataset.num_features}")

    return train_loader, valid_loader, test_loader, dataset.num_features


def load_bbbp(root="./data", batch_size=32):
    return load_dataset(name="BBBP", root=root, batch_size=batch_size)


def load_bace(root="./data", batch_size=32):
    return load_dataset(name="BACE", root=root, batch_size=batch_size)


if __name__ == "__main__":
    load_bbbp()
    load_bace()
