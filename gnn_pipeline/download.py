import io
import os
import zipfile

import requests

ZIP_URL = "https://github.com/bstabler/TransportationNetworks/archive/master.zip"
DEFAULT_EXTRACT_DIR = "TransportationNetworks_master"


def ensure_dataset(extract_dir: str = DEFAULT_EXTRACT_DIR, network: str = "Philadelphia") -> str:
    """Lädt den TransportationNetworks-Datensatz, falls er noch nicht lokal liegt.

    Gibt den Pfad zum Ordner des gewünschten Netzwerks zurück
    (z. B. TransportationNetworks_master/TransportationNetworks-master/Philadelphia).
    """
    network_dir = os.path.join(extract_dir, "TransportationNetworks-master", network)
    if os.path.isdir(network_dir):
        print(f"Datensatz bereits vorhanden: {network_dir}")
        return network_dir

    print(f"Downloading data from: {ZIP_URL}")
    response = requests.get(ZIP_URL)
    response.raise_for_status()

    print("Extracting data...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(extract_dir)
    print(f"Data extracted to: {extract_dir}")

    if not os.path.isdir(network_dir):
        raise FileNotFoundError(f"Netzwerk-Ordner nicht gefunden: {network_dir}")
    return network_dir
