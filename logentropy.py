import torch
from torch.nn import Module
import torch.nn.functional as F


class LogEntropyLoss(Module):
    # a loss for subset targets data
    def __init__(self):
        super(LogEntropyLoss, self).__init__()

    def forward(self, logits, target):
        # logits format:[[a,b,c,d],[e,f,g,h]]
        # target format:[[1,1,1,0],[0,1,1,0]]
        logits = logits.contiguous()  # [NHW, C]
        logits = F.softmax(logits, 1)
        mask = target.bool()
        logits = logits.masked_fill(~mask, 0)
        loss = torch.sum(logits, dim=1)
        loss = torch.log(loss)
        loss = -1 * loss
        return(loss.mean())

