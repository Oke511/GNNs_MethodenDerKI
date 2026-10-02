"""Gemeinsamer Trainingsloop für Link Prediction.

Erwartete Modell-Schnittstelle (wie GATv2LinkPredictor):
    model(x, edge_index, edge_attr) -> Knoten-Embeddings
    model.predict_link(node_embeddings, edge_label_index) -> Logits pro Kantenpaar
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data


@torch.no_grad()
def evaluate_link_predictor(model, train: Data, split: Data, loss_fn=None) -> tuple[float, float]:
    """Embeddings aus dem Trainingsgraphen, bewertet auf split.edge_label_index."""
    loss_fn = loss_fn or BCEWithLogitsLoss()
    model.eval()
    emb = model(train.x, train.edge_index, train.edge_attr)
    scores = model.predict_link(emb, split.edge_label_index)
    labels = split.edge_label.float()
    loss = loss_fn(scores, labels).item()
    auc = roc_auc_score(labels.cpu().numpy(), torch.sigmoid(scores).cpu().numpy())
    return loss, auc


def train_link_predictor(
    model,
    train: Data,
    val: Data,
    test: Data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    log_every: int = 10,
) -> dict[str, list[float]]:
    """Full-Batch-Training, gibt History mit train_loss, val_auc, test_auc zurück."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "test_loss": [], "test_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        emb = model(train.x, train.edge_index, train.edge_attr)
        scores = model.predict_link(emb, train.edge_label_index)
        loss = loss_fn(scores, train.edge_label.float())
        loss.backward()
        optimizer.step()

        val_loss, val_auc = evaluate_link_predictor(model, train, val, loss_fn)
        test_loss, test_auc = evaluate_link_predictor(model, train, test, loss_fn)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["test_loss"].append(test_loss)
        history["test_auc"].append(test_auc)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d} | train loss {loss.item():.4f} | val AUC {val_auc:.4f} | test AUC {test_auc:.4f}")

    best = max(range(epochs), key=lambda i: history["val_auc"][i])
    print(
        f"\nBeste Val-AUC {history['val_auc'][best]:.4f} in Epoche {best + 1} "
        f"(Test-AUC dort: {history['test_auc'][best]:.4f}); finale Test-AUC {history['test_auc'][-1]:.4f}"
    )
    return history


def plot_auc_history(history: dict[str, list[float]], title: str = "ROC-AUC über Epochen"):
    epochs = range(1, len(history["val_auc"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["val_auc"], label="Validation ROC-AUC")
    ax.plot(epochs, history["test_auc"], label="Test ROC-AUC")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig
