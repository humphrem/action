"""
This module contains the implementation of the MegadetectorDetector class, which
inherits from the BaseDetector class. It uses the Megadetector ONNX model for 
object detection and includes functions for non-max suppression and post-processing 
of the model's outputs.
"""

import numpy as np
from onnxruntime.capi._pybind_state import get_available_providers

from .base_detector import BaseDetector

# The Megadetector ONNX model should be in the ./models directory and needs to be pulled with git-lfs
megadetector_model_path = "models/md_v5a_1_3_640_640_static.onnx"
# Input images should be 640 x 640
megadetector_image_width = 640
megadetector_image_height = 640


# Code inspired by https://github.com/parlaynu/megadetector-v5-onnx, used under MIT License
# https://github.com/parlaynu/megadetector-v5-onnx/blob/main/LICENSE


def _calc_ious(b0, bx):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        b0 (numpy array): First bounding box.
        bx (numpy array): Second bounding box.

    Returns:
        float: IoU value.
    """
    # intersection area: (max(0, min(bottom) - max(top))) * (max(0, min(right) - max(left)))
    i_area = np.maximum(
        np.minimum(b0[2:4], bx[..., 2:4]) - np.maximum(b0[:2], bx[..., :2]), 0
    ).prod(axis=1)

    # union area: area_b0 + area_bx - intersection area
    u_area = (
        (b0[2:4] - b0[:2]).prod(axis=0)
        + (bx[..., 2:4] - bx[..., :2]).prod(axis=-1)
        - i_area
    )

    return i_area / u_area


def _nms(pred, iou_thresh, npred):
    """
    Perform non-max suppression on the predictions.

    Args:
        pred (numpy array): Predictions from the model.
        iou_thresh (float): IoU threshold for suppression.
        npred (list): List to store the predictions after suppression.

    Returns:
        list: Predictions after non-max suppression.
    """
    if len(pred) == 0:
        return npred

    p0 = pred[0]
    px = pred[1:]

    npred.append(p0)
    if len(px) == 0:
        return npred

    # create the IOUs
    ious = _calc_ious(p0, px)

    # for non-matching class, set the iou to 0 so they don't get considered
    ious[px[..., 5] != p0[5]] = 0

    # get the remaining predictions
    pp = px[ious < iou_thresh]

    return _nms(pp, iou_thresh, npred)


def non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45):
    """
    Perform non-max suppression on the predictions and filter them based on
    confidence threshold.

    Args:
        pred (numpy array): Predictions from the model.
        conf_thresh (float, optional): Confidence threshold for filtering. Defaults to 0.25.
        iou_thresh (float, optional): IoU threshold for suppression. Defaults to 0.45.

    Returns:
        list: Filtered predictions after non-max suppression.
    """
    # prediction: x, y, w, h, box_conf, cls0_conf, cls2_cnf, ...
    # filter (by box conf_thresh) and sort the predictions
    pred = pred[pred[..., 4] > conf_thresh]
    pred = pred[np.flip(np.argsort(pred[..., 4], axis=-1), axis=0)]

    # replace class prob with class label
    pred[..., 5] = np.argmax(pred[..., 5:], axis=-1)
    pred = pred[..., :6]

    # convert boxes to xyxy
    pred[..., :4] = _xywh2xyxy(pred[..., :4])

    # run the nms
    return _nms(pred, iou_thresh, [])


def _xywh2xyxy(xywh):
    """
    Convert bounding box format from center x, center y, width, height to
    top-left x, top-left y, bottom-right x, bottom-right y.

    Args:
        xywh (numpy array): Bounding box in center x, center y, width, height format.

    Returns:
        numpy array: Bounding box in top-left x, top-left y, bottom-right x, bottom-right y format.
    """
    xyxy = np.zeros_like(xywh)
    xc, yc, half_w, half_h = xywh[:, 0], xywh[:, 1], xywh[:, 2] / 2, xywh[:, 3] / 2
    xyxy[:, 0] = xc - half_w
    xyxy[:, 1] = yc - half_h
    xyxy[:, 2] = xc + half_w
    xyxy[:, 3] = yc + half_h
    return xyxy


class MegadetectorDetector(BaseDetector):
    """
    MegadetectorDetector class that inherits from BaseDetector. It uses the
    Megadetector ONNX model for object detection and includes methods for
    post-processing of the model's outputs.
    """

    def __init__(self, logger, min_duration, buffer, confidence):
        """
        Initialize the MegadetectorDetector class.

        Args:
            logger (Logger): Logger object for logging.
            min_duration (float): The minimum duration of a generated clip (defaults to 10.0)
            buffer (float): An optional number of seconds to add before/after a clip (defaults to 5.0)
            confidence (float): The confidence level to use (defaults to 0.50)

        """
        logger.info(
            "Initializing Megadetector Model and Optimizing (this will take a minute...)"
        )
        # Use some defaults if any of these aren't already set
        min_duration = 10.0 if min_duration is None else min_duration
        buffer = 5.0 if buffer is None else buffer
        confidence = 0.50 if confidence is None else confidence
        providers = get_available_providers()
        super().__init__(
            logger,
            megadetector_model_path,
            megadetector_image_width,
            megadetector_image_height,
            min_duration,
            buffer,
            confidence,
            "Animal",
            providers,
        )

    def post_processing(self, outputs):
        """
        Perform post-processing on the model's outputs. This includes non-max
        suppression, filtering based on confidence threshold, and conversion of
        absolute coordinates to relative.

        Args:
            outputs (numpy array): Outputs from the model.

        Returns:
            list: Post-processed predictions.
        """
        preds = []
        for p in outputs[0]:
            # TODO: what to use for IOU_THRESHOLD? Defaults to 0.45 in run-onnx.py
            p = non_max_suppression(p, self.confidence, 0.45)
            # Filter out predictions for animals (class=0)
            p = [pred for pred in p if pred[5] == 0]
            # If there are predictions, convert absolute coordinates to relative
            if len(p) > 0:
                p = np.array(p)
                p[..., :4] = p[..., :4] / [
                    self.input_image_width,
                    self.input_image_height,
                    self.input_image_width,
                    self.input_image_height,
                ]

            preds.append(p)

        return preds
