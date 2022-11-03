# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os


import numpy as np
import torch
import utils.network_losses as net_loss
from datetime import datetime as dt



def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x

def init_linear(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: torch.nn.init.zeros_(m.bias)

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())





def get_loss_function(cfg, input_loss):
    if input_loss.lower() == 'bceloss':
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif input_loss.lower() == 'iou':
        loss_func = net_loss.IoULoss()
    elif input_loss.lower() == 'crossentropyloss':
        label_weights = torch.tensor(cfg.CONST.LABEL_WEIGHTS).cuda()
        # loss_func = GeneralizedDiceLoss(include_background=False)
        loss_func = torch.nn.CrossEntropyLoss()
    elif input_loss.lower() == 'focalloss':
        loss_func = net_loss.FocalLoss()
    elif input_loss.lower() == 'tverskyloss':
        loss_func = net_loss.TverskyLoss()
    elif input_loss.lower() == 'focaltverskyloss':
        loss_func = net_loss.FocalTverskyLoss()
    else:
        raise Exception(
            '[FATAL] %s No matching loss function available for: %s. voxels' % (dt.now(), input_loss.NETWORK.LOSS_FUNC))
    return loss_func






def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
