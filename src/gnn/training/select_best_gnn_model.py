from src.gnn.models.registry import get_gnn_models


def select_best_gnn_model(
    results_df,
    dataset_name,
):

    dataset_df = results_df[results_df["dataset_name"] == dataset_name]

    gnn_models = set(get_gnn_models().keys())

    gnn_df = dataset_df[dataset_df["model"].isin(gnn_models)].copy()

    if gnn_df.empty:

        raise ValueError(f"No GNN models found for " f"{dataset_name}")

    best_index = gnn_df["pr_auc"].idxmax()

    return gnn_df.loc[best_index]
