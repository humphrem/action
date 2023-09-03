"""
This module contains the implementation of the YoloFishDetector class, which
inherits from the BaseDetector class. It uses the YOLO-Fish ONNX model for 
object detection and includes functions for non-max suppression and post-processing 
of the model's outputs.
"""

import numpy as np

from base_detector import BaseDetector

# Code inspired by https://github.com/Tianxiaomo/pytorch-YOLOv4, used under Apache-2.0 License
# https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/License.txt


# The YOLO-Fish ONNX model should be in the same directory as this file
yolo_fish_model_path = "yolov4_1_3_608_608_static.onnx"
# Input images should be 608 x 608
yolo_fish_image_width = 608
yolo_fish_image_height = 608


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.array): Array of bounding boxes.
        confs (np.array): Array of confidence scores for each box.
        nms_thresh (float, optional): Threshold for NMS. Defaults to 0.5.
        min_mode (bool, optional): If True, use minimum instead of union for IoU calculation. Defaults to False.

    Returns:
        np.array: Indices of boxes to keep after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


class YoloFishDetector(BaseDetector):
    """
    YoloFishDetector class that inherits from BaseDetector. It uses the
    YOLO-Fish ONNX model for object detection and includes methods for
    post-processing of the model's outputs.
    """

    def __init__(self, logger, min_duration, buffer, confidence):
        """
        Initialize the YoloFishDetector class.

        Args:
            logger (Logger): Logger object for logging.
            min_duration (float): The minimum duration of a generated clip (defaults to 3.0)
            buffer (float): An optional number of seconds to add before/after a clip (defaults to 1.0)
            confidence (float): The confidence level to use (defaults to 0.45)
        """
        logger.info("Initializing YOLO-Fish Model")
        # Use some defaults if any of these aren't already set
        min_duration = 3.0 if min_duration is None else min_duration
        buffer = 1.0 if buffer is None else buffer
        confidence = 0.45 if confidence is None else confidence
        super().__init__(
            logger,
            yolo_fish_model_path,
            yolo_fish_image_width,
            yolo_fish_image_height,
            min_duration,
            buffer,
            confidence,
            "Fish",
        )

    def post_processing(self, outputs):
        """
        Perform post-processing on the model outputs. This includes non-max
        suppression, and filtering based on confidence threshold.

        Args:
            outputs (list): List of model outputs.

        Returns:
            list: List of bounding boxes for detected objects.
        """
        # [batch, num, 1, 4]
        box_array = outputs[0]
        # [batch, num, num_classes]
        confs = outputs[1]

        # Normalize the shape of box_array and confs
        box_array = box_array[:, :, 0, :]
        confs = confs[:, :, 0]

        # Concatenate (join) the three arrays for coord, confidence, and class to form
        # bounding boxes of the form: [x1, y1, x2, y2, confidence, class]
        box_array = np.concatenate(
            (
                box_array,
                confs[..., None],
                # The class will always be 0 (i.e., fish)
                np.zeros((box_array.shape[0], box_array.shape[1], 1)),
            ),
            axis=2,
        )

        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()

        # Filter out overlapping bounding boxes using the non-max suppression function
        nms_threshold = 0.6
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = box_array[i, :, 4] > self.confidence
            filtered_boxes = box_array[i, argwhere, :]

            bboxes = []
            keep = nms_cpu(filtered_boxes[:, :4], filtered_boxes[:, 4], nms_threshold)

            if keep.size > 0:
                filtered_boxes = filtered_boxes[keep, :]

                for k in range(filtered_boxes.shape[0]):
                    bboxes.append(
                        [
                            filtered_boxes[k, 0],  # x1
                            filtered_boxes[k, 1],  # x2
                            filtered_boxes[k, 2],  # x2
                            filtered_boxes[k, 3],  # y2
                            filtered_boxes[k, 4],  # confidence
                            filtered_boxes[k, 5],  # class
                        ]
                    )

            bboxes_batch.append(bboxes)

        return bboxes_batch
