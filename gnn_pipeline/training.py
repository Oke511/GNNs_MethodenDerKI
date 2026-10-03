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


# Farben für die Learning Curves (farbfehlsicht-sicher, Reihenfolge fest): Train = blau, Val = orange, Test = aqua
_SERIES_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
_SURFACE, _INK, _MUTED, _GRID, _AXIS = "#ffffff", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def _style_axis(ax, xlabel: str, ylabel: str, n_epochs: int):
    # Farben explizit setzen: PyCharm rendert Notebook-Plots sonst mit dunklem Hintergrund/weißem Text
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.grid(True, color=_GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelcolor=_INK, length=0)
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    ax.set_xlim(1, n_epochs * 1.22)   # rechts Platz für Endlabels
    ax.set_xticks([t for t in ax.get_xticks() if 0 < t <= n_epochs])


def _label_line_ends(ax, series: list[tuple[str, float, str]], fmt: str, x_end: int):
    """Wert am Linienende beschriften; bei Kollision Labels vertikal auseinanderschieben, Leader-Linie verbindet."""
    to_frac = lambda v: ax.transAxes.inverted().transform(ax.transData.transform((x_end, v)))[1]
    min_gap = 0.07   # Achsenanteil
    ordered = sorted(series, key=lambda s: to_frac(s[1]))
    positions: list[float] = []
    for _, y, _ in ordered:
        f = to_frac(y)
        positions.append(f if not positions else max(f, positions[-1] + min_gap))
    overflow = positions[-1] - 0.97
    if overflow > 0:
        positions = [p - overflow for p in positions]
    for (name, y, color), f in zip(ordered, positions):
        ax.annotate(
            f"{name} {fmt.format(y)}",
            xy=(x_end, y), xycoords="data", xytext=(1.03, f), textcoords="axes fraction",
            va="center", ha="left", fontsize=9, color=_INK,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1, shrinkA=0, shrinkB=3),
            annotation_clip=False,
        )


def _mark_point(ax, x: float, y: float, color: str):
    ax.plot(x, y, "o", ms=8, color=color, markeredgecolor=_SURFACE, markeredgewidth=2, zorder=5)


def plot_learning_curves(history: dict[str, list[float]], title: str = "Learning Curve"):
    """Zwei Panels: links BCE-Loss (Train/Val/Test, log-Skala), rechts ROC-AUC (Val/Test).

    Die Epoche mit der besten Val-AUC ist in beiden Panels als senkrechte Linie markiert –
    daran lässt sich ablesen, ob Val-Loss und Val-AUC zusammenpassen (Overfitting, Instabilität).
    Log-Skala beim Loss, weil der Val-/Test-Loss in den ersten Epochen explodieren kann (GATv2: >100)
    und sonst den restlichen Verlauf plattdrückt.
    """
    n = len(history["train_loss"])
    epochs = list(range(1, n + 1))
    best = max(range(n), key=lambda i: history["val_auc"][i])
    best_ep = best + 1

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(15, 5.4))
    fig.patch.set_facecolor(_SURFACE)

    # --- Loss ---
    loss_series = [
        ("Train", "train_loss", _SERIES_COLORS["train"]),
        ("Val", "val_loss", _SERIES_COLORS["val"]),
        ("Test", "test_loss", _SERIES_COLORS["test"]),
    ]
    for label, key, color in loss_series:
        ax_loss.plot(epochs, history[key], color=color, linewidth=2, solid_joinstyle="round", label=f"{label}-Loss")
    ax_loss.set_yscale("log")
    _style_axis(ax_loss, "Epoche", "BCE-Loss (log)", n)
    ax_loss.set_title("Loss", loc="left", color=_INK, fontsize=12)
    _label_line_ends(ax_loss, [(l, history[k][-1], c) for l, k, c in loss_series], "{:.3f}", n)

    # --- ROC-AUC ---
    auc_series = [
        ("Val", "val_auc", _SERIES_COLORS["val"]),
        ("Test", "test_auc", _SERIES_COLORS["test"]),
    ]
    for label, key, color in auc_series:
        ax_auc.plot(epochs, history[key], color=color, linewidth=2, solid_joinstyle="round", label=f"{label}-ROC-AUC")
    y_min = min(min(history["val_auc"]), min(history["test_auc"]))
    ax_auc.set_ylim(max(0.0, 0.1 * int(y_min * 10) - 0.05), 1.01)
    _style_axis(ax_auc, "Epoche", "ROC-AUC", n)
    ax_auc.set_title("ROC-AUC", loc="left", color=_INK, fontsize=12)
    _label_line_ends(ax_auc, [(l, history[k][-1], c) for l, k, c in auc_series], "{:.3f}", n)

    # --- beste Val-Epoche in beiden Panels ---
    for ax in (ax_loss, ax_auc):
        ax.axvline(best_ep, color=_MUTED, linewidth=1, linestyle="-", alpha=0.8)
    _mark_point(ax_loss, best_ep, history["val_loss"][best], _SERIES_COLORS["val"])
    _mark_point(ax_auc, best_ep, history["val_auc"][best], _SERIES_COLORS["val"])
    ax_auc.annotate(
        f"beste Val-AUC {history['val_auc'][best]:.3f} in Epoche {best_ep}\n"
        f"Test-AUC dort {history['test_auc'][best]:.3f} · final {history['test_auc'][-1]:.3f}",
        xy=(best_ep, history["val_auc"][best]), xycoords="data",
        xytext=(0.97, 0.06), textcoords="axes fraction", ha="right", va="bottom", fontsize=9, color=_INK,
        arrowprops=dict(arrowstyle="-", color=_MUTED, linewidth=1, shrinkA=0, shrinkB=6),
    )

    for ax in (ax_loss, ax_auc):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9,
                  labelcolor=_INK)

    fig.suptitle(title, x=0.01, ha="left", color=_INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
