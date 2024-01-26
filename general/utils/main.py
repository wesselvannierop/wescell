import torch
import numpy as np


def remap(original: torch.IntTensor, map: dict) -> torch.tensor:
    """_summary_

    Args:
        original (torch.tensor): _description_
        map (dict): key (map_form) -> value (map_to)

    Returns:
        torch.tensor: items in original that are not in map will be zero
    """
    print("Remapping...")
    remapped = torch.zeros_like(original)
    for map_from, map_to in map.items():
        remapped[original == int(map_from)] = map_to
    return remapped


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN"""

    confusion_matrix_arrs = {}

    groundtruth = groundtruth.astype(np.uint8)
    predicted = predicted.astype(np.uint8)

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs["tp"] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs["tn"] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs["fp"] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs["fn"] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(groundtruth, predicted):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    confusion_matrix_colors = {
        "tp": (0, 255, 255),  # cyan
        "fp": (255, 0, 255),  # magenta
        "fn": (255, 255, 0),  # yellow
        "tn": (0, 0, 0),  # black
    }

    groundtruth = groundtruth.squeeze()
    predicted = predicted.squeeze()
    assert predicted.shape == groundtruth.shape

    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 3))
    for label, mask in masks.items():
        color = confusion_matrix_colors[label]
        mask_rgb = np.zeros_like(color_mask)
        mask_rgb[mask] = color
        color_mask += mask_rgb

    return color_mask


def remove_sacred_from_ckpt(path):
    """Most of the checkpoints saved by Wessel require the sacred library.
    This function needs that library aswell but saves a ckpt file that can be used without sacred.
    """
    x = torch.load(path, map_location="cpu")
    x.pop("hyper_parameters")
    torch.save(x, path + ".fixed")
