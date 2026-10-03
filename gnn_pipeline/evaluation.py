"""Einordnung der Modellergebnisse: Heuristik-Baselines und Fehleranalyse nach Link-Typ."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


def _undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    ei = edge_index.cpu().numpy()
    a = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(num_nodes, num_nodes)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def heuristic_scores(split: Data) -> dict[str, np.ndarray]:
    """Klassische Link-Prediction-Heuristiken für split.edge_label_index, berechnet auf dem
    Message-Passing-Graphen des Splits (ungerichtet) und den Rohkoordinaten split.pos."""
    u, v = (t.cpu().numpy() for t in split.edge_label_index)
    adj = _undirected_adjacency(split.edge_index, split.num_nodes)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_log_deg = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    pos = split.pos.cpu().numpy()
    return {
        "-Distanz (Koordinaten)": -np.linalg.norm(pos[u] - pos[v], axis=1),
        "Gemeinsame Nachbarn": np.asarray(adj[u].multiply(adj[v]).sum(axis=1)).ravel(),
        "Adamic-Adar": np.asarray(adj[u].multiply(adj[v] @ sp.diags(inv_log_deg)).sum(axis=1)).ravel(),
        "Niedriger Grad (Loch)": -(deg[u] + deg[v]),
    }


def heuristic_baselines(train: Data, val: Data, test: Data) -> pd.DataFrame:
    """ROC-AUC der Heuristiken je Split. Zeigt, wie schwer die Aufgabe ohne Lernen ist.

    - Hohe AUC für "-Distanz": Negative liegen räumlich weit auseinander (leichte Negative).
    - Hohe AUC für "Niedriger Grad (Loch)": Die entfernte Label-Kante hinterlässt an ihren Endpunkten einen
      geringeren Grad, Negative haben das nicht - ein Modell kann Löcher statt Straßenstruktur erkennen.
    Mit neg_strategy="hole" sollten beide Werte nahe 0.5 liegen.
    """
    rows = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        labels = split.edge_label.cpu().numpy()
        rows[name] = {h: roc_auc_score(labels, s) for h, s in heuristic_scores(split).items()}
    return pd.DataFrame(rows).rename_axis("Heuristik")


@torch.no_grad()
def auc_by_link_type(model, split: Data, edge_attr_columns: list[str]) -> pd.DataFrame:
    """Fehleranalyse: ROC-AUC der positiven Kanten eines Link-Typs gegen alle Negativen des Splits.

    Der Link-Typ der Positiven kommt aus split.edge_label_attr (echte Features der Kante). Typ 7 mit
    Kapazität 999999 sind die künstlichen Zonen-Konnektoren.
    """
    model.eval()
    emb = model(split.x, split.edge_index, split.edge_attr)
    scores = torch.sigmoid(model.predict_link(emb, split.edge_label_index)).cpu().numpy()
    labels = split.edge_label.cpu().numpy()
    type_cols = [i for i, c in enumerate(edge_attr_columns) if c.startswith("link_type_")]
    attr = split.edge_label_attr.cpu().numpy()
    link_type = np.array([edge_attr_columns[type_cols[j]] for j in attr[:, type_cols].argmax(axis=1)])

    neg = labels == 0
    rows = []
    for t in sorted(set(link_type[labels == 1])):
        pos = (labels == 1) & (link_type == t)
        mask = pos | neg
        rows.append({
            "link_type": t,
            "positive Kanten": int(pos.sum()),
            "ROC-AUC vs. alle Negativen": roc_auc_score(labels[mask], scores[mask]),
            "mittlere Wahrscheinlichkeit": float(scores[pos].mean()),
        })
    return pd.DataFrame(rows).set_index("link_type")
