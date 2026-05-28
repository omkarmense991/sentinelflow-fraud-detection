import torch

from torch_geometric.data import Data

from sklearn.model_selection import train_test_split

from src.gnn.data.load_elliptic_dataset import load_elliptic_dataset


def build_graph_dataset():

    features_df, classes_df, edges_df = load_elliptic_dataset()

    # =========================================
    # Rename Columns
    # =========================================

    features_df = features_df.rename(columns={0: "txId"})

    # =========================================
    # Merge Labels
    # =========================================

    df = features_df.merge(classes_df, on="txId")

    # =========================================
    # Remove Unknown Labels
    # =========================================

    df = df[df["class"] != "unknown"].copy()

    # =========================================
    # Encode Labels
    # illicit = 1
    # licit = 0
    # =========================================

    df["label"] = df["class"].map({"1": 1, "2": 0})

    # =========================================
    # Node Mapping
    # =========================================

    node_mapping = {tx_id: idx for idx, tx_id in enumerate(df["txId"])}

    # =========================================
    # Filter Valid Edges
    # =========================================

    edges_df = edges_df[
        edges_df["txId1"].isin(node_mapping) & edges_df["txId2"].isin(node_mapping)
    ]

    # =========================================
    # Build edge_index
    # =========================================

    edge_index = torch.tensor(
        [
            [node_mapping[src] for src in edges_df["txId1"]],
            [node_mapping[dst] for dst in edges_df["txId2"]],
        ],
        dtype=torch.long,
    )

    # =========================================
    # Node Features
    # =========================================

    feature_columns = [
        col for col in df.columns if col not in ["txId", "class", "label"]
    ]

    x = torch.tensor(df[feature_columns].values, dtype=torch.float)

    # =========================================
    # Labels
    # =========================================

    y = torch.tensor(df["label"].values, dtype=torch.long)

    # =========================================
    # Create Train/Val/Test Masks
    # =========================================

    num_nodes = len(df)

    indices = torch.arange(num_nodes)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, stratify=y[train_idx], random_state=42
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True

    val_mask[val_idx] = True

    test_mask[test_idx] = True

    # =========================================
    # Build Data Object
    # =========================================

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data
