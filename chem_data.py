"""Read official splits or make one reproducible split from the supplied CSV."""
from pathlib import Path
import csv
from collections import defaultdict
import random
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.utils.smiles import from_smiles

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def canonical(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def read_split(path):
    """Return rows in exactly the organizer's order, with no dropped samples."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')
        if reader.fieldnames is None:
            raise ValueError(f'{path}: empty or missing header')
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        smi_key = fields.get('smiles')
        label_key = next((fields[k] for k in ('label', 'bbb+/bbb-', 'y') if k in fields), None)
        if smi_key is None or label_key is None:
            raise ValueError(f'{path}: expected columns SMILES and label (or BBB+/BBB- or y)')
        result = []
        for number, row in enumerate(reader, 2):
            smi = canonical(row.get(smi_key))
            label = str(row.get(label_key, '')).strip().upper()
            if smi is None or label not in ('0', '1', 'BBB-', 'BBB+'):
                raise ValueError(f'{path}:{number}: invalid SMILES or BBB+/BBB- label; no rows may be silently dropped')
            result.append((smi, int(label in ('1', 'BBB+'))))
    if not result:
        raise ValueError(f'{path}: no labelled molecules')
    return result


def load_fixed_splits(data_dir):
    folder = Path(data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f'{folder}: expected train/valid/test CSV files')
    paths = {}
    for key in ('train', 'valid', 'test'):
        candidates = [folder / (key + suffix) for suffix in ('.csv', '.tsv')]
        present = [p for p in candidates if p.is_file()]
        if len(present) != 1:
            raise FileNotFoundError(f'Expected exactly one of {candidates[0]} or {candidates[1]}')
        paths[key] = present[0]
    parts = {key: read_split(path) for key, path in paths.items()}
    # Keep the provided split intact; fail if canonical duplicate structures cross it.
    sets = {key: {s for s, _ in records} for key, records in parts.items()}
    for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f'Fixed splits overlap ({left}/{right}: {len(overlap)} exact canonical structures). '
                             'Do not use these results until the split source is resolved.')
    for key, records in parts.items():
        if len(records) < 2 or len({y for _, y in records}) != 2:
            raise ValueError(f'{key} must contain both BBB classes to report ROC-AUC')
        seen = {}
        for smi, label in records:
            if smi in seen and seen[smi] != label:
                raise ValueError(f'{key} has contradictory labels for {smi}')
            seen[smi] = label
    return parts, paths


def generate_scaffold_splits(source, output_dir, seed=42):
    """Make a fixed, documented 80/10/10 scaffold split from a supplied raw CSV.

    This is a project split, NOT an organizer-provided split. Invalid structures
    and contradictory duplicate structures are removed before splitting.
    """
    if seed != 42:
        raise ValueError('Hackathon seed is fixed at 42')
    source = Path(source)
    with source.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {'SMILES', 'label'} <= set(reader.fieldnames):
            raise ValueError(f'{source}: expected SMILES,label columns')
        records = list(reader)
    by_structure = defaultdict(set)
    invalid = 0
    for row in records:
        smi = canonical(row['SMILES'])
        label = row['label'].strip()
        if smi is None or label not in ('0', '1'):
            invalid += 1
            continue
        by_structure[smi].add(int(label))
    conflicting = sum(len(values) > 1 for values in by_structure.values())
    clean = [(smi, next(iter(labels))) for smi, labels in by_structure.items() if len(labels) == 1]
    if len(clean) < 30:
        raise ValueError('Too few valid, non-conflicting molecules to make fixed splits')
    groups = defaultdict(list)
    for smi, label in clean:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        # Ring-free compounds have no Murcko scaffold; keep distinct structures separate.
        groups[scaffold or smi].append((smi, label))
    chunks = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(chunks)
    chunks.sort(key=len, reverse=True)  # stable sort retains seeded order for ties
    parts = {'train': [], 'valid': [], 'test': []}
    train_max = int(.8 * len(clean))
    valid_max = int(.1 * len(clean))
    for group in chunks:
        if len(parts['train']) + len(group) <= train_max:
            key = 'train'
        elif len(parts['valid']) + len(group) <= valid_max:
            key = 'valid'
        else:
            key = 'test'
        parts[key].extend(group)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for key, values in parts.items():
        with (output_dir / f'{key}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SMILES', 'label'])
            writer.writerows(values)
    # Re-read and verify exactly the same paths used by every model.
    load_fixed_splits(output_dir)
    return {'raw_rows': len(records), 'invalid_rows': invalid,
            'conflicting_canonical_structures': conflicting,
            'duplicate_or_conflicting_rows_removed': len(records) - len(clean),
            'unique_molecules': len(clean), 'scaffold_groups': len(groups),
            'counts': {key: len(values) for key, values in parts.items()},
            'method': 'seed 42 Bemis-Murcko scaffold groups; self-generated, not organizer-provided'}


def fingerprint(smiles):
    arr = np.zeros(1024, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles)), arr)
    return arr


def rd_fp(smiles):
    return _GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))


def graph(smiles, label=None):
    item = from_smiles(smiles)
    item.x = item.x.float()
    item.smiles = smiles
    if label is not None:
        item.y = torch.tensor([float(label)], dtype=torch.float32)
    return item
