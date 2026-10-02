"""Parser für die .tntp-Dateien (Knoten, Kanten, Trips)."""

import os
from dataclasses import dataclass

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


@dataclass
class NetworkFrames:
    """Rohdaten eines Netzwerks als DataFrames.

    nodes: node_id, x, y, type ('zone' | 'crossing')
    edges: from_node, to_node, Capacity, Length, free_flow_time, B, Power, Speed_Limit, Toll, Type
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame


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
                continue
    return pd.DataFrame(node_data)


def parse_zone_nodes(trips_file_path: str) -> set[int]:
    """Alle Origin-/Destination-Knoten der Trips-Datei gelten als Zonen."""
    zone_nodes = set()
    in_origin_block = False
    with open(trips_file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(";"):
                continue
            if stripped.startswith("Origin"):
                in_origin_block = True
                try:
                    zone_nodes.add(int(stripped.split(" ")[1]))
                except (ValueError, IndexError):
                    pass
            elif in_origin_block:
                parts = stripped.split(" :")
                if parts and parts[0].strip().isdigit():
                    zone_nodes.add(int(parts[0].strip()))
                else:
                    in_origin_block = False
    return zone_nodes


def parse_edges(net_file_path: str) -> pd.DataFrame:
    network_data = []
    header_columns: list[str] = []
    found_header = False

    with open(net_file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(";"):
                continue

            if stripped.startswith("~") and "init_node" in stripped and not found_header:
                header_str = stripped.replace("~", "").strip()
                raw_parts = [c.strip() for c in header_str.split("\t")]
                filtered = [p for p in raw_parts if p and p != ";"]
                header_columns = [COLUMN_NAME_MAP.get(c.lower(), c) for c in filtered]
                found_header = True
                continue

            if found_header and stripped:
                raw_parts = [p.strip() for p in stripped.split("\t")]
                parts = [p for p in raw_parts if p and p != ";"]
                if len(parts) != len(header_columns):
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
                    continue

    if not network_data:
        raise ValueError(f"Keine Kantendaten in {net_file_path} gefunden.")
    return pd.DataFrame(network_data)


def load_network(network_dir: str, name: str = "Philadelphia") -> NetworkFrames:
    """Liest Knoten, Zonen und Kanten eines Netzwerks ein."""
    nodes_df = parse_nodes(os.path.join(network_dir, f"{name}_node.tntp"))
    zone_nodes = parse_zone_nodes(os.path.join(network_dir, f"{name}_trips.tntp"))
    nodes_df["type"] = nodes_df["node_id"].apply(lambda n: "zone" if n in zone_nodes else "crossing")

    edges_df = parse_edges(os.path.join(network_dir, f"{name}_net.tntp"))

    print(f"{name}: {len(nodes_df)} Knoten ({len(zone_nodes)} Zonen), {len(edges_df)} Kanten")
    return NetworkFrames(nodes=nodes_df, edges=edges_df)
