"""Bausteine, die alle Modell-Notebooks teilen, damit sich die Modelle nur im Message Passing unterscheiden."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def batch_norm(num_features: int) -> nn.BatchNorm1d:
    """BatchNorm ohne Running-Stats: normalisiert immer mit den Statistiken des aktuellen Graphen.

    Bei Full-Batch-Training sieht jeder Forward-Pass alle Knoten; Training und Auswertung rechnen auf
    demselben Graphen, die Normalisierung ist dadurch in beiden Modi identisch. Mit Running-Stats wichen
    eval- und train-Modus voneinander ab, was die Loss-Kurven springen ließ.
    """
    return nn.BatchNorm1d(num_features, track_running_stats=False)


class EdgeAwareInput(nn.Module):
    """Eingangsschicht: h_i = W_x x_i + sum_{j->i} W_e e_ji.

    Mit nur (x, y) als Knotenfeature sind die Koordinaten benachbarter Knoten fast identisch. Ein
    Attention-Mittelwert über Nachbarn (GATv2) liefert dann kaum mehr als die eigene Position, und die
    Kantenfeatures wirken dort nur auf die Gewichte, nicht auf die Nachricht - GATv2 blieb bei AUC 0.5.
    Diese Schicht bringt Kantenfeatures und lokale Struktur in die Knotenrepräsentation, für beide Modelle gleich.
    """

    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int):
        super().__init__()
        self.lin_x = nn.Linear(in_channels, hidden_channels)
        self.lin_e = nn.Linear(edge_dim, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        incoming = scatter(self.lin_e(edge_attr), edge_index[1], dim=0, dim_size=x.size(0), reduce="sum")
        return self.lin_x(x) + incoming


class LinkDecoder(nn.Module):
    """MLP auf concat(h_src, h_dst) -> ein Logit pro Paar. Richtungsabhängig (A->B != B->A).

    Nutzt bewusst keine Features der Zielkante, sonst wäre die Vorhersage trivial.
    """

    def __init__(self, embedding_dim: int, hidden_channels: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = node_embeddings[edge_label_index[0]], node_embeddings[edge_label_index[1]]
        return self.mlp(torch.cat([src, dst], dim=1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
