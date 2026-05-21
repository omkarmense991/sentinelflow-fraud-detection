# src/models/model_selector.py

def select_best_model(results_df, min_precision=0.80):

    filtered_df = results_df[results_df["cv_precision_mean"] >= min_precision]

    if filtered_df.empty:

        raise ValueError(
            f"No models satisfy minimum precision " f"requirement of {min_precision}"
        )

    best_index = filtered_df["cv_pr_auc_mean"].idxmax()

    best_model = filtered_df.loc[best_index]

    return best_model
