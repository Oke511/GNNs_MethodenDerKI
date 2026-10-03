"""Gemeinsame Datenpipeline und Trainingshilfen für die Link-Prediction-Notebooks.

Aufbau:
    download.py   - Netz- und Knotendatei (TransportationNetworks) laden
    parsing.py    - .tntp-Dateien in DataFrames überführen (Zonen laut Dateikopf)
    features.py   - Feature Engineering für Kanten und Knoten (erweiterbar über Registry)
    graph.py      - PyG Data-Objekt bauen, Split erzeugen, Kantenfeatures an Splits hängen
    io.py         - fertigen Split speichern/laden
    models.py     - gemeinsame Bausteine: Eingangsschicht, Decoder, BatchNorm
    training.py   - Trainingsloop, Seeds, Mehr-Seed-Auswertung, Learning-Curve-Plots
    evaluation.py - Heuristik-Baselines und Fehleranalyse nach Link-Typ
"""

from .download import ensure_dataset, DEFAULT_EXTRACT_DIR
from .parsing import load_network, parse_metadata, NetworkFrames
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
from .models import EdgeAwareInput, LinkDecoder, batch_norm, count_parameters
from .training import (
    set_seed,
    train_link_predictor,
    evaluate_link_predictor,
    evaluate_splits,
    run_seeds,
    plot_auc_history,
    plot_learning_curves,
)
from .evaluation import heuristic_baselines, auc_by_link_type

__all__ = [
    "ensure_dataset",
    "DEFAULT_EXTRACT_DIR",
    "load_network",
    "parse_metadata",
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
    "EdgeAwareInput",
    "LinkDecoder",
    "batch_norm",
    "count_parameters",
    "set_seed",
    "train_link_predictor",
    "evaluate_link_predictor",
    "evaluate_splits",
    "run_seeds",
    "plot_auc_history",
    "plot_learning_curves",
    "heuristic_baselines",
    "auc_by_link_type",
]
