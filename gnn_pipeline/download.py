import os

import requests

RAW_URL = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/{network}/{network}_{kind}.tntp"
DEFAULT_EXTRACT_DIR = "TransportationNetworks_master"
REQUIRED_FILES = ("net", "node")


def ensure_dataset(extract_dir: str = DEFAULT_EXTRACT_DIR, network: str = "Philadelphia", timeout: float = 60) -> str:
    """Lädt die benötigten .tntp-Dateien eines Netzwerks, falls sie noch nicht lokal liegen.

    Statt des kompletten Repository-ZIPs werden nur Netz- und Knotendatei geladen. Die Ordnerstruktur
    entspricht dem entpackten ZIP, ein bereits vorhandener Download wird also weiterverwendet.
    Gibt den Pfad zum Ordner des Netzwerks zurück
    (z. B. TransportationNetworks_master/TransportationNetworks-master/Philadelphia).
    """
    network_dir = os.path.join(extract_dir, "TransportationNetworks-master", network)
    missing = [kind for kind in REQUIRED_FILES
               if not os.path.isfile(os.path.join(network_dir, f"{network}_{kind}.tntp"))]
    if not missing:
        print(f"Datensatz bereits vorhanden: {network_dir}")
        return network_dir

    os.makedirs(network_dir, exist_ok=True)
    for kind in missing:
        url = RAW_URL.format(network=network, kind=kind)
        target = os.path.join(network_dir, f"{network}_{kind}.tntp")
        print(f"Downloading {url}")
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with open(target, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    print(f"Daten liegen in: {network_dir}")
    return network_dir
