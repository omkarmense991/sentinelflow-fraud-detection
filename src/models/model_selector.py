# src/tabular_ml/models/model_selector.py


def select_best_model(results_df, dataset_name, min_precision=0.80):

    # =========================================
    # Dataset Filter
    # =========================================

    dataset_df = results_df[results_df["dataset_name"] == dataset_name]

    # =========================================
    # Precision Filter
    # =========================================

    filtered_df = dataset_df[dataset_df["cv_precision_mean"] >= min_precision]

    if filtered_df.empty:

        raise ValueError(
            f"No models satisfy "
            f"minimum precision "
            f"requirement of "
            f"{min_precision}"
        )

    best_index = filtered_df["pr_auc"].idxmax()

    best_model = filtered_df.loc[best_index]

    return best_model
