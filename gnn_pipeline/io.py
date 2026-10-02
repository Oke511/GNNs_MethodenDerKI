"""Speichern/Laden des fertigen Splits, damit alle Modell-Notebooks dieselben Daten sehen."""

import os
from dataclasses import asdict

import torch

from .graph import GraphBundle

DEFAULT_BUNDLE_PATH = os.path.join("processed", "philadelphia_split.pt")


def save_bundle(bundle: GraphBundle, path: str = DEFAULT_BUNDLE_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(asdict(bundle), path)
    print(f"Split gespeichert: {path}")
    return path


def load_bundle(path: str = DEFAULT_BUNDLE_PATH, device: torch.device | str | None = None) -> GraphBundle:
    """Lädt den Split. PyG-Data-Objekte sind gepickelt, daher weights_only=False."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} nicht gefunden. Zuerst data_prep.ipynb ausführen.")
    payload = torch.load(path, weights_only=False)
    bundle = GraphBundle(**payload)
    if device is not None:
        bundle.data = bundle.data.to(device)
        bundle.train = bundle.train.to(device)
        bundle.val = bundle.val.to(device)
        bundle.test = bundle.test.to(device)
    print(f"Split geladen: {path}\n{bundle.summary()}")
    return bundle
