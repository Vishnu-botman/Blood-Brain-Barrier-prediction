import argparse
import compat  # noqa: F401  (patches gcn_norm for torch 2.13)
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from data import load_dataset
from models import MODELS


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        y_true.extend(batch.y.squeeze(-1).tolist())
        y_score.extend(torch.sigmoid(out).tolist())
    return roc_auc_score(y_true, y_score)


def run(model_name, dataset_name="BBBP", n_layers=2, hidden_dim=64, epochs=150, lr=0.001,
        batch_size=32, seed=42, device="cpu", verbose=True):
    torch.manual_seed(seed)
    train_loader, valid_loader, test_loader, in_dim = load_dataset(name=dataset_name, batch_size=batch_size, seed=seed)

    model = MODELS[model_name](in_dim, hidden_dim, 1, n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_auc = evaluate(model, valid_loader, device)
        test_auc = evaluate(model, test_loader, device)
        if val_auc > best_val:
            best_val, best_test = val_auc, test_auc
        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d} loss {train_loss:.4f} "
                  f"val {val_auc:.4f} test {test_auc:.4f}")
    print(f"[{dataset_name.upper()}] {model_name} ({n_layers} layers) best val {best_val:.4f} test {best_test:.4f}", flush=True)
    return best_val, best_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="gcn")
    parser.add_argument("--dataset", choices=["bbbp", "bace"], default="bbbp")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.model, dataset_name=args.dataset, n_layers=args.layers, hidden_dim=args.hidden,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch, seed=args.seed)
