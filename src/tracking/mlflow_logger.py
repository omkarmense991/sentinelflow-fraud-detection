import mlflow


def log_experiment(params, metrics):

    mlflow.log_params(params)

    mlflow.log_metrics(metrics)
