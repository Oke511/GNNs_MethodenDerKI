"""Parser für die .tntp-Dateien (Metadaten, Knoten, Kanten)."""

import os
import re
from dataclasses import dataclass, field

import pandas as pd

# Standard-TNTP-Spaltennamen -> im Projekt verwendete Namen
COLUMN_NAME_MAP = {
    "init_node": "from_node",
    "term_node": "to_node",
    "capacity": "Capacity",
    "length": "Length",
    "free_flow_time": "free_flow_time",
    "b": "B",
    "power": "Power",
    "speed": "Speed_Limit",
    "toll": "Toll",
    "link_type": "Type",
}
INT_COLUMNS = {"from_node", "to_node", "Type"}
FLOAT_COLUMNS = {"Capacity", "Length", "free_flow_time", "B", "Power", "Speed_Limit", "Toll"}

_METADATA_PATTERN = re.compile(r"^<([^>]+)>\s*(.*)$")


@dataclass
class NetworkFrames:
    """Rohdaten eines Netzwerks als DataFrames.

    nodes: node_id, x, y, type ('zone' | 'crossing')
    edges: from_node, to_node, Capacity, Length, free_flow_time, B, Power, Speed_Limit, Toll, Type
    metadata: Kopfzeilen der Netzdatei, z. B. {"NUMBER OF ZONES": "1525", ...}
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    metadata: dict[str, str] = field(default_factory=dict)


def parse_metadata(tntp_file_path: str) -> dict[str, str]:
    """Liest den Metadaten-Kopf (<KEY> value) bis <END OF METADATA>."""
    metadata = {}
    with open(tntp_file_path, "r") as f:
        for line in f:
            match = _METADATA_PATTERN.match(line.strip())
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            if key == "END OF METADATA":
                break
            metadata[key] = value
    return metadata


def parse_nodes(node_file_path: str) -> pd.DataFrame:
    node_data = []
    with open(node_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                node_data.append({"node_id": int(parts[0]), "x": float(parts[1]), "y": float(parts[2])})
            except ValueError:
                continue  # Kopfzeile ("Node X Y ;")
    return pd.DataFrame(node_data)


def parse_edges(net_file_path: str) -> pd.DataFrame:
    network_data = []
    header_columns: list[str] = []
    found_header = False
    skipped = 0

    with open(net_file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(";"):
                continue

            if stripped.startswith("~") and "init_node" in stripped and not found_header:
                header_str = stripped.replace("~", "").strip()
                filtered = [p for p in header_str.split() if p != ";"]
                header_columns = [COLUMN_NAME_MAP.get(c.lower(), c) for c in filtered]
                found_header = True
                continue

            if found_header and stripped:
                parts = [p for p in stripped.split() if p != ";"]
                if len(parts) != len(header_columns):
                    skipped += 1
                    continue
                try:
                    row = {}
                    for col, val in zip(header_columns, parts):
                        if col in INT_COLUMNS:
                            row[col] = int(float(val))
                        elif col in FLOAT_COLUMNS:
                            row[col] = float(val)
                        else:
                            row[col] = val
                    network_data.append(row)
                except ValueError:
                    skipped += 1

    if not network_data:
        raise ValueError(f"Keine Kantendaten in {net_file_path} gefunden.")
    if skipped:
        print(f"Warnung: {skipped} Kantenzeilen in {net_file_path} nicht lesbar und übersprungen.")
    return pd.DataFrame(network_data)


def load_network(network_dir: str, name: str = "Philadelphia") -> NetworkFrames:
    """Liest Knoten und Kanten eines Netzwerks ein und prüft die Anzahlen gegen den Dateikopf.

    Zonen: Nach TNTP-Konvention sind die Knoten 1..<NUMBER OF ZONES> die Zonen (Origin/Destination,
    <FIRST THRU NODE> = Zonenanzahl + 1). Früher wurden Zonen aus der Trips-Datei geparst; das hat
    Zonen ohne Trips und alle Origins (mehrere Leerzeichen in "Origin       1") übersehen.
    """
    net_path = os.path.join(network_dir, f"{name}_net.tntp")
    metadata = parse_metadata(net_path)
    num_zones = int(metadata["NUMBER OF ZONES"])

    nodes_df = parse_nodes(os.path.join(network_dir, f"{name}_node.tntp"))
    nodes_df["type"] = (nodes_df["node_id"] <= num_zones).map({True: "zone", False: "crossing"})

    edges_df = parse_edges(net_path)

    for key, actual in [("NUMBER OF NODES", len(nodes_df)), ("NUMBER OF LINKS", len(edges_df))]:
        expected = int(metadata.get(key, actual))
        if expected != actual:
            raise ValueError(f"{name}: {key} laut Dateikopf {expected}, eingelesen {actual}.")

    n_zones = int((nodes_df["type"] == "zone").sum())
    print(f"{name}: {len(nodes_df)} Knoten ({n_zones} Zonen), {len(edges_df)} Kanten")
    return NetworkFrames(nodes=nodes_df, edges=edges_df, metadata=metadata)
