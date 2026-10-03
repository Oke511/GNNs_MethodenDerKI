"""PyG Data-Objekt bauen, in Train/Val/Test splitten und Kantenfeatures an die Splits hängen.

Split-Strategie (siehe `split_graph`):
    Straßennetze sind fast vollständig reziprok (A->B und B->A existieren beide). Ein Split, der
    gerichtete Kanten unabhängig voneinander zieht (`RandomLinkSplit(is_undirected=False)`), legt
    für ~77 % der Val-/Test-Kanten die Gegenrichtung in den Trainingsgraphen. Das Modell sieht
    B->A beim Message Passing und soll A->B "vorhersagen" - ein Leak, der die AUC künstlich nach
    oben treibt. Standard ist deshalb der paarweise Split (`pair_aware=True`): Beide Richtungen
    eines Knotenpaars landen immer im selben Split. `reverse_edge_leakage` misst den Leak.
"""

import math
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from .features import FeatureConfig, build_edge_features, build_node_features, engineer_edge_features
from .parsing import NetworkFrames


@dataclass
class GraphBundle:
    """Alles, was die Modell-Notebooks brauchen – und nichts aus den Rohdaten."""

    data: Data            # vollständiger Graph (für Embedding-Visualisierung)
    train: Data           # edge_index/edge_attr = Message-Passing-Kanten, edge_label_* = Trainingspaare
    val: Data
    test: Data
    node_feature_columns: list[str]
    edge_attr_columns: list[str]
    node_type: list[str]  # 'zone' | 'crossing' pro PyG-Knotenindex
    original_node_ids: list[int]
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Knoten: {self.data.num_nodes}, Kanten: {self.data.edge_index.size(1)}",
            f"Knotenfeatures ({len(self.node_feature_columns)}): {self.node_feature_columns}",
            f"Kantenfeatures ({len(self.edge_attr_columns)}): {self.edge_attr_columns}",
        ]
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            pos = int((split.edge_label == 1).sum())
            neg = int((split.edge_label == 0).sum())
            lines.append(f"{name}: msg-passing Kanten={split.edge_index.size(1)}, Label-Paare pos={pos} neg={neg}")
        for name, split in [("val", self.val), ("test", self.test)]:
            leaked, total = reverse_edge_leakage(self.train, split)
            lines.append(
                f"{name}: positive Kanten mit Gegenrichtung im Train-Graph: {leaked}/{total} ({leaked / max(total, 1):.1%})"
            )
        if self.meta:
            lines.append(f"meta: {self.meta}")
        return "\n".join(lines)


def build_graph(frames: NetworkFrames, config: FeatureConfig) -> tuple[Data, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Feature Engineering + Umwandlung in ein PyG Data-Objekt.

    Gibt (data, edges_df, nodes_df_sorted, node_feature_columns, edge_attr_columns) zurück.
    edges_df enthält die engineerten Spalten, nodes_df_sorted ist nach PyG-Index sortiert.
    """
    nodes_df = frames.nodes.copy()
    edges_df = engineer_edge_features(frames.edges, nodes_df, config)

    # Knoten-IDs auf 0..N-1 abbilden (nur Knoten, die in Kanten vorkommen – wie bisher)
    all_nodes = pd.concat([edges_df["from_node"], edges_df["to_node"]]).unique()
    all_nodes.sort()
    node_id_mapping = {orig: new for new, orig in enumerate(all_nodes)}

    edges_df["from_node_mapped"] = edges_df["from_node"].map(node_id_mapping)
    edges_df["to_node_mapped"] = edges_df["to_node"].map(node_id_mapping)
    edge_index = torch.tensor(edges_df[["from_node_mapped", "to_node_mapped"]].values.T, dtype=torch.long)

    edge_matrix, edge_attr_columns = build_edge_features(edges_df, config)
    edge_attr = torch.tensor(edge_matrix, dtype=torch.float)

    nodes_df = nodes_df[nodes_df["node_id"].isin(node_id_mapping)].copy()
    nodes_df["node_id_mapped"] = nodes_df["node_id"].map(node_id_mapping)
    nodes_df_sorted = nodes_df.sort_values("node_id_mapped").reset_index(drop=True)
    if len(nodes_df_sorted) != len(all_nodes):
        raise ValueError("Knotendatei und Kantendatei passen nicht zusammen (fehlende Knoten).")

    node_matrix, node_feature_columns = build_node_features(nodes_df_sorted, edges_df, config)
    x = torch.tensor(node_matrix, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(all_nodes))
    return data, edges_df, nodes_df_sorted, node_feature_columns, edge_attr_columns


# ---------------------------------------------------------------------------
# Schlüssel-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Gerichteter Schlüssel: (u, v) und (v, u) sind verschieden."""
    return edge_index[0] * num_nodes + edge_index[1]


def _pair_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Richtungsunabhängiger Schlüssel: (u, v) und (v, u) fallen auf (min, max) zusammen."""
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    return lo * num_nodes + hi


def reverse_edge_leakage(train: Data, split: Data) -> tuple[int, int]:
    """Zählt positive Label-Kanten (u->v) in `split`, deren Gegenrichtung (v->u) im
    Train-Message-Passing-Graphen liegt. Gibt (geleakt, gesamt) zurück; beim paarweisen Split ist geleakt == 0."""
    num_nodes = train.num_nodes
    pos = split.edge_label_index[:, split.edge_label == 1]
    reverse_keys = _edge_keys(pos.flip(0), num_nodes)
    train_keys = _edge_keys(train.edge_index, num_nodes)
    leaked = int(torch.isin(reverse_keys, train_keys).sum())
    return leaked, int(pos.size(1))


def _lookup_edge_attr(
    data: Data, query_index: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features für Kantenpaare: echte Features für existierende Kanten,
    für Negativ-Beispiele zufällig gezogene Features aus dem Originalgraphen
    (gleiches Verhalten wie im ursprünglichen Notebook, nur vektorisiert)."""
    num_nodes = data.num_nodes
    keys = _edge_keys(data.edge_index, num_nodes)
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    q = _edge_keys(query_index, num_nodes)
    pos = torch.searchsorted(sorted_keys, q).clamp(max=len(sorted_keys) - 1)
    found = sorted_keys[pos] == q

    out = torch.empty(query_index.size(1), data.edge_attr.size(1), dtype=data.edge_attr.dtype)
    out[found] = data.edge_attr[order[pos[found]]]
    n_missing = int((~found).sum())
    if n_missing:
        rand = torch.randint(0, data.edge_attr.size(0), (n_missing,), generator=generator)
        out[~found] = data.edge_attr[rand]
    return out, found


# ---------------------------------------------------------------------------
# Split-Varianten
# ---------------------------------------------------------------------------

def _sample_negative_pairs(
    existing_pair_keys: torch.Tensor, num_nodes: int, num_pairs: int, generator: torch.Generator
) -> torch.Tensor:
    """Zieht `num_pairs` verschiedene ungeordnete Knotenpaare {u, v}, u != v, die im Graphen in
    keiner Richtung existieren. Rejection Sampling: bei 13k Knoten und ~20k Paaren liegt die
    Trefferquote nahe 100 %, daher reichen ein bis zwei Runden."""
    collected = []
    unique_keys = torch.empty(0, dtype=torch.long)
    while unique_keys.numel() < num_pairs:
        n_draw = int((num_pairs - unique_keys.numel()) * 1.2) + 16
        u = torch.randint(0, num_nodes, (n_draw,), generator=generator)
        v = torch.randint(0, num_nodes, (n_draw,), generator=generator)
        keep = u != v
        keys = torch.minimum(u, v)[keep] * num_nodes + torch.maximum(u, v)[keep]
        collected.append(keys[~torch.isin(keys, existing_pair_keys)])
        unique_keys = torch.unique(torch.cat(collected))
    # torch.unique sortiert - wieder mischen, damit die Zuordnung zu den Splits zufällig bleibt
    unique_keys = unique_keys[torch.randperm(unique_keys.numel(), generator=generator)[:num_pairs]]
    return torch.stack([unique_keys // num_nodes, unique_keys % num_nodes])


def _split_graph_pairwise(
    data: Data, seed: int, val_ratio: float, test_ratio: float, neg_sampling_ratio: float
) -> tuple[Data, Data, Data]:
    """Paarweiser Split: Train/Val/Test werden über ungeordnete Knotenpaare {u, v} gezogen,
    beide Richtungen einer Straße erben denselben Split. Verhindert den reziproken Leak."""
    generator = torch.Generator().manual_seed(seed)
    num_nodes = data.num_nodes

    pair_keys = _pair_keys(data.edge_index, num_nodes)
    unique_pairs, pair_of_edge = torch.unique(pair_keys, return_inverse=True)
    num_pairs = unique_pairs.numel()

    # 0 = train, 1 = val, 2 = test - pro Knotenpaar zugewiesen, an beide Richtungen vererbt
    perm = torch.randperm(num_pairs, generator=generator)
    n_val = int(round(val_ratio * num_pairs))
    n_test = int(round(test_ratio * num_pairs))
    pair_split = torch.zeros(num_pairs, dtype=torch.long)
    pair_split[perm[:n_val]] = 1
    pair_split[perm[n_val:n_val + n_test]] = 2
    edge_split = pair_split[pair_of_edge]

    # Positive Label: die tatsächlich existierenden gerichteten Kanten (Einbahnstraßen nur einmal)
    positives = {s: data.edge_index[:, edge_split == s] for s in (0, 1, 2)}

    # Negative Label: Paare, die in keiner Richtung existieren, jeweils in beide Richtungen
    # eingetragen. So bleibt neg/pos ~= neg_sampling_ratio und Negative werden symmetrisch behandelt.
    n_neg_pairs = {s: math.ceil(neg_sampling_ratio * positives[s].size(1) / 2) for s in (0, 1, 2)}
    neg_pairs = _sample_negative_pairs(unique_pairs, num_nodes, sum(n_neg_pairs.values()), generator)
    negatives, offset = {}, 0
    for s in (0, 1, 2):
        chunk = neg_pairs[:, offset:offset + n_neg_pairs[s]]
        offset += n_neg_pairs[s]
        negatives[s] = torch.cat([chunk, chunk.flip(0)], dim=1)

    def make_split(msg_mask: torch.Tensor, s: int) -> Data:
        label_index = torch.cat([positives[s], negatives[s]], dim=1)
        label = torch.cat([torch.ones(positives[s].size(1)), torch.zeros(negatives[s].size(1))])
        return Data(
            x=data.x,
            edge_index=data.edge_index[:, msg_mask],
            edge_attr=data.edge_attr[msg_mask],
            num_nodes=num_nodes,
            edge_label_index=label_index,
            edge_label=label,
        )

    # Message-Passing-Graphen wie bei RandomLinkSplit: train/val sehen die Train-Kanten, test zusätzlich die Val-Kanten
    train = make_split(edge_split == 0, 0)
    val = make_split(edge_split == 0, 1)
    test = make_split(edge_split != 2, 2)
    return train, val, test


def _split_graph_directed(
    data: Data, seed: int, val_ratio: float, test_ratio: float, neg_sampling_ratio: float
) -> tuple[Data, Data, Data]:
    """Bisheriges Verhalten: RandomLinkSplit über gerichtete Kanten (reziproker Leak!).
    Nur noch zum Vergleich / zur Reproduktion alter Ergebnisse."""
    # negative_sampling in PyG nutzt Pythons random-Modul, daher alle RNGs seeden
    seed_everything(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train, val, test = transform(data)

    generator = torch.Generator().manual_seed(seed)
    train_attr, found = _lookup_edge_attr(data, train.edge_index, generator)
    if not bool(found.all()):
        raise RuntimeError("Train-Message-Passing-Kanten nicht im Originalgraph gefunden.")
    train.edge_attr = train_attr
    return train, val, test


def split_graph(
    data: Data,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_sampling_ratio: float = 2.0,
    pair_aware: bool = True,
) -> tuple[Data, Data, Data]:
    """Train/Val/Test-Split mit festem Seed, anschließend Kantenfeatures an die Splits hängen.

    pair_aware=True (Standard): paarweiser Split, beide Richtungen eines Knotenpaars im selben Split
                                (siehe Modul-Docstring und `_split_graph_pairwise`).
    pair_aware=False:           alter kantenweiser RandomLinkSplit mit reziprokem Leak.

    Rückgabe für beide Varianten gleich:
    - train.edge_attr: Features der Message-Passing-Kanten (Teilgraph)
    - *.edge_label_attr: Features zu edge_label_index (Negativ-Beispiele: zufällig gezogen)
    - val/test.edge_attr werden entfernt, um Verwechslung zu vermeiden
    """
    if pair_aware:
        train, val, test = _split_graph_pairwise(data, seed, val_ratio, test_ratio, neg_sampling_ratio)
    else:
        train, val, test = _split_graph_directed(data, seed, val_ratio, test_ratio, neg_sampling_ratio)

    generator = torch.Generator().manual_seed(seed + 1)
    for split in (train, val, test):
        split.x = data.x
        split.edge_label_attr, _ = _lookup_edge_attr(data, split.edge_label_index, generator)
    for split in (val, test):
        if "edge_attr" in split:
            del split.edge_attr
    return train, val, test
