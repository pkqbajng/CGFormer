import torch
import torch.nn as nn
def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean