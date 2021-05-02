import torch
from losses import cob_loss, lcfcn_loss

def getLoss(loss_name):
    if loss_name == "BCELoss":
        criterion = torch.nn.BCELoss()
    elif loss_name == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == "cob":
        criterion = cob_loss.computeLoss
    elif loss_name == "lcfcn":
        criterion = lcfcn_loss.computeLoss
    else:
        raise ValueError(f"Loss {loss_name} not implemented.")

    return criterion