import torch
from .vocab import IDX2CHAR


def ctc_greedy_decode(logits):
    """
    logits: T x num_classes
    """
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    prev_idx = -1
    output = []
    for idx in preds:
        if idx != prev_idx and idx != 0:
            output.append(IDX2CHAR[idx])
        prev_idx = idx
    return "".join(output)
