import importlib

import numpy as np
import torch
from skimage import measure
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error
from utils.helpers import expand_as_one_hot


def MeanIoU(input, target):
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    assert input.dim() == 5

    n_classes = input.size()[1]

    if target.dim() == 4:
        target = expand_as_one_hot(target, C=n_classes)

    assert input.size() == target.size()

    per_batch_iou = []
    per_channel_iou_total = []
    for _input, _target in zip(input, target):
        binary_prediction = binarize_predictions(_input, n_classes)

        # convert to uint8 just in case
        binary_prediction = binary_prediction.byte()
        _target = _target.byte()

        per_channel_iou_batch = []
        for c in range(n_classes):
            per_channel_iou_batch.append(jaccard_index(binary_prediction[c], _target[c]).cpu().numpy())

        assert per_channel_iou_batch, "All channels were ignored from the computation"
        mean_iou = np.mean(per_channel_iou_batch)
        per_batch_iou.append(mean_iou)
        per_channel_iou_total.append(per_channel_iou_batch)

    return torch.mean(torch.tensor(per_batch_iou)).cpu(), per_channel_iou_total


def binarize_predictions(input_volume, n_classes):
    """
    Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
    same size as the input tensor.
    """
    if n_classes == 1:
        # for single channel input just threshold the probability map
        result = input_volume > 0.5
        return result.long()

    _, max_index = torch.max(input_volume, dim=0, keepdim=True)
    return torch.zeros_like(input_volume, dtype=torch.uint8).scatter_(0, max_index, 1)


def jaccard_index(prediction, target):
    """
    Computes IoU for a given target and prediction tensors
    """
    return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


def percentage_diff(pred_val, gt_val):
    return ((pred_val - gt_val) / gt_val) * 100


def calc_biomarkers(cfg, pred_volume, ground_truth_volume=None):
    pred_volume = pred_volume.cpu().numpy()
    if ground_truth_volume is not None:
        ground_truth_volume = ground_truth_volume.cpu().numpy()

    pred_volume = np.argmax(pred_volume, axis=1)
    lv_myocardium = np.sum(pred_volume == cfg.LABELS.LV) / (cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
    rv_myocardium = np.sum(pred_volume == cfg.LABELS.RV) / (cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
    la_myocardium = np.sum(pred_volume == cfg.LABELS.LA) / (cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
    ra_myocardium = np.sum(pred_volume == cfg.LABELS.RA) / (cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)

    if ground_truth_volume is not None:
        lv_myocardium_gt = np.sum(ground_truth_volume == cfg.LABELS.LV) / (
                cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
        rv_myocardium_gt = np.sum(ground_truth_volume == cfg.LABELS.RV) / (
                cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
        la_myocardium_gt = np.sum(ground_truth_volume == cfg.LABELS.LA) / (
                cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)
        ra_myocardium_gt = np.sum(ground_truth_volume == cfg.LABELS.RA) / (
                cfg.LABELS.METADATA.VOXEL_SCALING ** 3 * 1000)

        lv_percent_diff = percentage_diff(lv_myocardium, lv_myocardium_gt)
        rv_percent_diff = percentage_diff(rv_myocardium, rv_myocardium_gt)
        la_percent_diff = percentage_diff(la_myocardium, la_myocardium_gt)
        ra_percent_diff = percentage_diff(ra_myocardium, ra_myocardium_gt)
        myocardium_vals = [lv_myocardium, rv_myocardium, la_myocardium, ra_myocardium]
        myocardium_diffs = [lv_percent_diff, rv_percent_diff, la_percent_diff, ra_percent_diff]
        return myocardium_vals, myocardium_diffs
    else:
        return [lv_myocardium, rv_myocardium, la_myocardium, ra_myocardium]
