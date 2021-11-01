# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):

        real_answers = real_answers[:, 1:]
        answer_mask = torch.ne(real_answers, 2)
        
        y_pred = pred_answers[answer_mask].float()
        y_true = real_answers[answer_mask].float()
        
        loss=nn.BCELoss()(y_pred, y_true)
        return loss, y_pred, y_true


def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)