"""PyG Data-Objekt bauen, in Train/Val/Test splitten und Kantenfeatures an die Splits hängen."""

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


def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edge_index[0] * num_nodes + edge_index[1]


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


def split_graph(
    data: Data,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_sampling_ratio: float = 2.0,
) -> tuple[Data, Data, Data]:
    """RandomLinkSplit mit festem Seed, anschließend Kantenfeatures korrekt an Splits hängen.

    - train.edge_attr: Features der Message-Passing-Kanten (Teilgraph)
    - *.edge_label_attr: Features zu edge_label_index (Negativ-Beispiele: zufällig gezogen)
    - val/test.edge_attr werden entfernt, um Verwechslung zu vermeiden
    """
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

    for split in (train, val, test):
        split.x = data.x
        split.edge_label_attr, _ = _lookup_edge_attr(data, split.edge_label_index, generator)
    for split in (val, test):
        if "edge_attr" in split:
            del split.edge_attr
    return train, val, test
