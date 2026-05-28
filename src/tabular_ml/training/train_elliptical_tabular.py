from src.tabular_ml.training.tabular_trainer import run_training

from src.tabular_ml.data.load_elliptic_tabular import load_elliptic_tabular_dataset

from src.tabular_ml.features.elliptical_preprocessing import split_elliptic_data

run_training(
    dataset_name="elliptic",
    dataset_loader=load_elliptic_tabular_dataset,
    split_function=split_elliptic_data,
)
