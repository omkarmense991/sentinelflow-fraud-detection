from tabular_ml.training.tabular_trainer import run_training

from tabular_ml.data.load_credit_card_data import load_credit_card_dataset

from tabular_ml.features.preprocessing import split_data

run_training(
    dataset_name="creditcard",
    dataset_loader=load_credit_card_dataset,
    split_function=split_data,
)
