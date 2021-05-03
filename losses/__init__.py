import torch
from losses import cob_loss, point_loss

def getLoss(loss_name):
    if loss_name == "BCELoss":
        criterion = torch.nn.BCELoss()
    elif loss_name == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == "point_cob":
        criterion = cob_loss.computeLoss
    elif loss_name == "point":
        criterion = point_loss.computeLoss
    else:
        raise ValueError(f"Loss {loss_name} not implemented.")

    return criterion