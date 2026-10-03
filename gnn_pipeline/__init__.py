"""Gemeinsame Datenpipeline und Trainingshilfen für die Link-Prediction-Notebooks.

Aufbau:
    download.py  - Datensatz (TransportationNetworks) laden/entpacken
    parsing.py   - .tntp-Dateien in DataFrames überführen
    features.py  - Feature Engineering für Kanten und Knoten (erweiterbar über Registry)
    graph.py     - PyG Data-Objekt bauen, Split erzeugen, Kantenfeatures an Splits hängen
    io.py        - fertigen Split speichern/laden
    training.py  - Trainingsloop + Auswertung + Learning-Curve-Plots, identisch für alle Modelle
"""

from .download import ensure_dataset, DEFAULT_EXTRACT_DIR
from .parsing import load_network, NetworkFrames
from .features import (
    FeatureConfig,
    NODE_FEATURE_BUILDERS,
    register_node_feature,
    engineer_edge_features,
    build_node_features,
    build_edge_features,
)
from .graph import build_graph, split_graph, reverse_edge_leakage, GraphBundle
from .io import save_bundle, load_bundle
from .training import train_link_predictor, evaluate_link_predictor, plot_auc_history, plot_learning_curves

__all__ = [
    "ensure_dataset",
    "DEFAULT_EXTRACT_DIR",
    "load_network",
    "NetworkFrames",
    "FeatureConfig",
    "NODE_FEATURE_BUILDERS",
    "register_node_feature",
    "engineer_edge_features",
    "build_node_features",
    "build_edge_features",
    "build_graph",
    "split_graph",
    "GraphBundle",
    "reverse_edge_leakage",
    "save_bundle",
    "load_bundle",
    "train_link_predictor",
    "evaluate_link_predictor",
    "plot_auc_history",
    "plot_learning_curves",
]
