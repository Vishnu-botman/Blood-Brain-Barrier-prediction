"""
Visualizing GAT Attention Weights on Molecular Structures.
Extracts multi-head attention coefficients and maps them to atom-level importance scores.
"""

import compat  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from data import load_dataset
from models import GAT


def get_atom_attention(model, mol_data, device="cpu"):
    """
    Pass a single molecule through the GAT model and aggregate attention
    weights to compute an importance score for each atom.
    """
    model.eval()
    mol_data = mol_data.to(device)
    x = mol_data.x.float()
    edge_index = mol_data.edge_index
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    with torch.no_grad():
        out, attns = model(x, edge_index, batch=batch, return_attention_weights=True)
        prob = torch.sigmoid(out).item()

    # Aggregate layer 1 attention weights
    ei, alpha = attns[0]
    alpha_mean = alpha.mean(dim=-1).cpu().numpy()  # Average over heads
    ei = ei.cpu().numpy()

    num_atoms = x.size(0)
    atom_in_attention = np.zeros(num_atoms)
    
    # Sum incoming attention weights per atom
    for k in range(ei.shape[1]):
        target_atom = ei[1, k]
        atom_in_attention[target_atom] += alpha_mean[k]

    # Normalize scores between 0 and 1
    if atom_in_attention.max() > atom_in_attention.min():
        atom_norm = (atom_in_attention - atom_in_attention.min()) / (atom_in_attention.max() - atom_in_attention.min())
    else:
        atom_norm = np.ones(num_atoms)

    return prob, atom_in_attention, atom_norm


def run_visualization():
    print("Loading BBBP dataset for attention visualization...")
    train_loader, valid_loader, test_loader, in_dim = load_dataset("BBBP", batch_size=32)

    device = "cpu"
    model = GAT(in_dim, hidden_dim=64, out_dim=1, n_layers=2, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Training GAT model (30 epochs) to extract learned attention...")
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

    pos_mol = None
    neg_mol = None
    for batch in test_loader:
        for i in range(len(batch.y)):
            label = batch.y[i].item()
            smiles = batch.smiles[i]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and 10 <= mol.GetNumAtoms() <= 28:
                if label == 1.0 and pos_mol is None:
                    pos_mol = batch[i]
                elif label == 0.0 and neg_mol is None:
                    neg_mol = batch[i]
            if pos_mol is not None and neg_mol is not None:
                break
        if pos_mol is not None and neg_mol is not None:
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GAT Attention Weight Analysis: What Functional Groups Predict BBB Permeability?", fontsize=14, fontweight="bold")

    for idx, (m_data, title) in enumerate([
        (pos_mol, "BBB+ Penetrant (Passes Barrier)"),
        (neg_mol, "BBB- Non-Penetrant (Blocked by Barrier)")
    ]):
        prob, raw_attn, norm_attn = get_atom_attention(model, m_data)
        mol = Chem.MolFromSmiles(m_data.smiles)
        atom_labels = [f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()]

        # Subplot 1: RDKit 2D depiction
        ax_img = axes[idx, 0]
        img = Draw.MolToImage(mol, size=(450, 350))
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(f"{title}\nTrue Label: {int(m_data.y.item())} | GAT Pred: {prob:.3f}", fontsize=11, fontweight="bold")

        # Subplot 2: Atom Attention Bar Chart
        ax_bar = axes[idx, 1]
        top3_idx = np.argsort(norm_attn)[-3:]
        bar_colors = ["#e67e22" if i in top3_idx else "#3498db" for i in range(len(atom_labels))]

        bars = ax_bar.bar(range(len(atom_labels)), norm_attn, color=bar_colors, edgecolor="black", alpha=0.85)
        ax_bar.set_xticks(range(len(atom_labels)))
        ax_bar.set_xticklabels(atom_labels, rotation=45, ha="right", fontsize=9)
        ax_bar.set_ylabel("Normalized Attention", fontsize=10)
        ax_bar.set_title(f"Atom Attention Scores (Orange = Top 3 Attended Atoms)", fontsize=11)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_png = "gat_attention_analysis.png"
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"\n[SUCCESS] Attention analysis plot saved to {output_png}")


if __name__ == "__main__":
    run_visualization()
