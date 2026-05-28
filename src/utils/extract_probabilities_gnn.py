import torch.nn.functional as F


def extract_probabilities(logits, labels, mask):

    logits = logits[mask]

    labels = labels[mask]

    probabilities = F.softmax(logits, dim=1)[:, 1]

    probabilities_np = probabilities.detach().cpu().numpy()

    labels_np = labels.detach().cpu().numpy()

    return (labels_np, probabilities_np)
