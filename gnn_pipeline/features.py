"""Feature Engineering für Kanten und Knoten.

Knotenfeatures sind über eine Registry erweiterbar: Jeder Builder bekommt
(nodes_df, edges_df) und gibt einen DataFrame mit einer Zeile pro Knoten
(in Reihenfolge von nodes_df) zurück. Welche Builder verwendet werden,
steuert `FeatureConfig.node_features`.

Neues Knotenfeature hinzufügen:

    @register_node_feature("mein_feature")
    def _mein_feature(nodes_df, edges_df):
        return pd.DataFrame({"mein_feature": ...}, index=nodes_df.index)

und anschließend "mein_feature" in FeatureConfig.node_features aufnehmen.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CAPACITY_PLACEHOLDER = 999999.0

NodeFeatureBuilder = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
NODE_FEATURE_BUILDERS: dict[str, NodeFeatureBuilder] = {}


def register_node_feature(name: str):
    def decorator(fn: NodeFeatureBuilder) -> NodeFeatureBuilder:
        NODE_FEATURE_BUILDERS[name] = fn
        return fn

    return decorator


@dataclass
class FeatureConfig:
    """Steuert, welche Features in data.x und data.edge_attr landen."""

    # Namen aus NODE_FEATURE_BUILDERS; Reihenfolge = Spaltenreihenfolge in data.x
    node_features: list[str] = field(default_factory=lambda: ["coords"])
    # Standardisierung aller Knotenfeatures (StandardScaler über den gesamten Graphen)
    scale_node_features: bool = True

    # Kantenfeatures
    edge_numerical: list[str] = field(
        default_factory=lambda: ["Capacity", "Length", "free_flow_time", "Toll", "capacity_ratio", "delta_x", "delta_y"]
    )
    edge_binary: list[str] = field(default_factory=lambda: ["capacity_is_placeholder", "has_reciprocal"])
    one_hot_link_type: bool = True
    # log1p vor dem Skalieren: Kapazität, Länge usw. sind stark rechtsschief
    log_edge_numerical: bool = True
    scale_edge_numerical: bool = True
    drop_edge_columns: list[str] = field(default_factory=lambda: ["B", "Power", "Speed_Limit"])


# ---------------------------------------------------------------------------
# Kantenfeatures
# ---------------------------------------------------------------------------

def engineer_edge_features(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Platzhalter-Flag, One-Hot-Link-Type, Reziprok-Verhältnis, Koordinatendifferenzen.

    Die Platzhalter-Kapazität 999999 (Zonen-Konnektoren) wird durch 0 ersetzt, die Information trägt
    `capacity_is_placeholder`. Sonst dominiert der Platzhalter die Skalierung, und alle echten
    Kapazitäten landen nach dem StandardScaler praktisch auf demselben Wert.
    """
    edges_df = edges_df.copy()

    is_placeholder = edges_df["Capacity"] == CAPACITY_PLACEHOLDER
    edges_df["capacity_is_placeholder"] = is_placeholder.astype(int)
    real_capacity = edges_df["Capacity"].where(~is_placeholder)   # NaN für Platzhalter
    edges_df["Capacity"] = real_capacity.fillna(0.0)

    if config.one_hot_link_type and "Type" in edges_df.columns:
        edges_df = pd.get_dummies(edges_df, columns=["Type"], prefix="link_type")

    edges_df = edges_df.drop(columns=config.drop_edge_columns, errors="ignore")

    # Reziproke Kapazität: Capacity(A->B) / Capacity(B->A) auf echten Kapazitäten,
    # 0 falls keine Gegenkante existiert oder eine der beiden Kapazitäten ein Platzhalter ist
    edges_df["_real_capacity"] = real_capacity.values
    reciprocal = edges_df[["from_node", "to_node", "_real_capacity"]].rename(
        columns={"from_node": "to_node", "to_node": "from_node", "_real_capacity": "recip_capacity"}
    )
    edges_df = pd.merge(edges_df, reciprocal, on=["from_node", "to_node"], how="left", indicator=True)
    edges_df["has_reciprocal"] = (edges_df["_merge"] == "both").astype(int)
    edges_df["capacity_ratio"] = (edges_df["_real_capacity"] / edges_df["recip_capacity"]).fillna(0.0)
    edges_df = edges_df.drop(columns=["_real_capacity", "recip_capacity", "_merge"])

    # Absolute Koordinatendifferenzen zwischen Start- und Zielknoten
    coords = nodes_df.set_index("node_id")[["x", "y"]]
    edges_df["delta_x"] = (coords.loc[edges_df["to_node"], "x"].values - coords.loc[edges_df["from_node"], "x"].values)
    edges_df["delta_y"] = (coords.loc[edges_df["to_node"], "y"].values - coords.loc[edges_df["from_node"], "y"].values)
    edges_df["delta_x"] = edges_df["delta_x"].abs()
    edges_df["delta_y"] = edges_df["delta_y"].abs()

    return edges_df


def build_edge_features(edges_df: pd.DataFrame, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Wählt die Kantenspalten aus und gibt (Matrix, Spaltennamen) zurück."""
    ohe_cols = [c for c in edges_df.columns if c.startswith("link_type_")] if config.one_hot_link_type else []
    columns = config.edge_numerical + config.edge_binary + ohe_cols

    missing = [c for c in columns if c not in edges_df.columns]
    if missing:
        raise KeyError(f"Kantenfeatures fehlen in edges_df: {missing}")

    matrix = edges_df[columns].astype(float).values
    idx = [columns.index(c) for c in config.edge_numerical]
    if config.log_edge_numerical and idx:
        if (matrix[:, idx] < 0).any():
            raise ValueError("log_edge_numerical erwartet nicht-negative numerische Kantenfeatures.")
        matrix[:, idx] = np.log1p(matrix[:, idx])
    if config.scale_edge_numerical and idx:
        matrix[:, idx] = StandardScaler().fit_transform(matrix[:, idx])
    return matrix, columns


# ---------------------------------------------------------------------------
# Knotenfeatures (Registry)
# ---------------------------------------------------------------------------

@register_node_feature("coords")
def _coords(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Koordinaten x, y (werden mit scale_node_features=True standardisiert)."""
    return nodes_df[["x", "y"]].copy()


@register_node_feature("node_type")
def _node_type(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Binäres Flag: 1 = Zone (Origin/Destination), 0 = Kreuzung."""
    return pd.DataFrame({"is_zone": (nodes_df["type"] == "zone").astype(float).values}, index=nodes_df.index)


@register_node_feature("degree")
def _degree(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """In- und Out-Grad (log1p), berechnet auf dem vollständigen Graphen.

    Hinweis: Grad-Features basieren auf allen Kanten, auch denen, die später
    in Val/Test landen. Für eine strikt leakage-freie Variante müsste der Grad
    nur auf den Trainingskanten berechnet werden.
    """
    out_deg = edges_df["from_node"].value_counts()
    in_deg = edges_df["to_node"].value_counts()
    return pd.DataFrame(
        {
            "log_in_degree": np.log1p(nodes_df["node_id"].map(in_deg).fillna(0).values),
            "log_out_degree": np.log1p(nodes_df["node_id"].map(out_deg).fillna(0).values),
        },
        index=nodes_df.index,
    )


@register_node_feature("capacity_stats")
def _capacity_stats(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Summe der ein- und ausgehenden Kapazität pro Knoten (log1p, Platzhalter ignoriert)."""
    real = edges_df[edges_df["Capacity"] != CAPACITY_PLACEHOLDER]
    cap_out = real.groupby("from_node")["Capacity"].sum()
    cap_in = real.groupby("to_node")["Capacity"].sum()
    return pd.DataFrame(
        {
            "log_capacity_in": np.log1p(nodes_df["node_id"].map(cap_in).fillna(0).values),
            "log_capacity_out": np.log1p(nodes_df["node_id"].map(cap_out).fillna(0).values),
        },
        index=nodes_df.index,
    )


def build_node_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Setzt die in config.node_features gewählten Builder zusammen.

    nodes_df muss bereits in der Reihenfolge der PyG-Knotenindizes sortiert sein.
    Gibt (Matrix [num_nodes, num_features], Spaltennamen) zurück.
    """
    unknown = [n for n in config.node_features if n not in NODE_FEATURE_BUILDERS]
    if unknown:
        raise KeyError(f"Unbekannte Knotenfeatures {unknown}. Verfügbar: {sorted(NODE_FEATURE_BUILDERS)}")
    if not config.node_features:
        raise ValueError("Mindestens ein Knotenfeature ist erforderlich.")

    parts = [NODE_FEATURE_BUILDERS[name](nodes_df, edges_df) for name in config.node_features]
    frame = pd.concat(parts, axis=1)
    matrix = frame.astype(float).values
    if config.scale_node_features:
        matrix = StandardScaler().fit_transform(matrix)
    return matrix, list(frame.columns)
