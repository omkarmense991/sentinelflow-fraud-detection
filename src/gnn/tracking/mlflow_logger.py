import mlflow


def log_gnn_metrics(metrics):

    for metric_name, value in metrics.items():

        mlflow.log_metric(metric_name, value)
