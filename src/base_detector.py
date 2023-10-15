"""
This module contains the BaseDetector class, which is used as the basis for the 
MegadetectorDetector and YoloFishDetector classes.  The BaseDetector wraps an
ONNX model for object detection and includes functions for loading a model,
doing detections, and drawing those detections. The rest of the code for each
model is in the derived class (see yolo_fish_detector.py and megadetector_detector.py)
"""

import os
import time

import cv2
import numpy as np
import onnxruntime


class BaseDetector:
    """
    Base class for object detection models. See YoloFishDetector and
    MegadetectorDetector classes for actual implementations of these
    models.
    """

    def __init__(
        self,
        logger,
        model_path,
        input_image_height,
        input_image_width,
        min_duration,
        buffer,
        confidence,
        class_name,
        providers=None,
    ):
        """
        Initialize the detector with model parameters and provider.

        Args:
            logger: Logger object for logging purposes.
            model_path: Path to the model file.
            input_image_height: Height of the input image.
            input_image_width: Width of the input image.
            min_duration (float): The minimum duration of a generated clip
            buffer (float): An optional number of seconds to add before/after a clip
            confidence (float): The confidence level to use
            class_name: Name of the class to be detected.
            providers: List of providers for ONNX runtime. Default is CPUExecutionProvider.
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.logger = logger
        self.model_path = model_path
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width
        self.min_duration = min_duration
        self.buffer = buffer
        self.confidence = confidence
        self.class_name = class_name
        self.providers = providers
        self.session = None

    def load(self):
        """
        Load the model from the provided path and try to optimize.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"The model file {self.model_path} was not found.")

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = onnxruntime.InferenceSession(
            self.model_path, providers=self.providers, sess_options=sess_options
        )

    def detect(self, image_src):
        """
        Detect objects in the provided image.

        Args:
            image_src: Source image for object detection.

        Returns:
            List of bounding boxes for detected objects.
        """
        # Pre-process the image to the appropriate size and format for the model
        resized = cv2.resize(
            image_src,
            (self.input_image_width, self.input_image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        # Run detection on the re-shaped image
        start_time = time.time()
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})
        end_time = time.time()
        self.logger.debug(f"Detection took {end_time - start_time}s")

        # Process detection outputs into bounding boxes
        boxes = self.post_processing(outputs)

        # Return boxes[0] if it exists, otherwise return an empty list
        return boxes[0] if boxes else []

    def draw_detections(self, img, boxes, title):
        """
        Draw bounding boxes on the image for detected objects and show
        in a window.

        Args:
            img: Image on which to draw bounding boxes.
            boxes: List of bounding boxes.
            title: Title for the image window.
        """
        img = np.copy(img)
        width = img.shape[1]
        height = img.shape[0]

        # Draw a box around each detection with confidence score
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            bbox_thick = int(0.4 * (height + width) / 600)

            # Use red so it stands out against background, with red colour from government redside dace illustration
            bgr = (28, 23, 212)

            confidence = box[4]
            msg = f"{self.class_name} {confidence:.4f}"
            text_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            top_left = (x1, y1)
            bottom_right = (top_left[0] + text_size[0], top_left[1] - text_size[1] - 10)

            self.logger.debug(
                f"{x1}, {x2} x {x2}, {y2} - {confidence} {self.class_name}"
            )

            cv2.rectangle(
                img, (x1 - 1, y1), (int(bottom_right[0]), int(bottom_right[1])), bgr, -1
            )
            img = cv2.putText(
                img,
                msg,
                (top_left[0], int(top_left[1] - 5)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

            img = cv2.rectangle(img, (x1, y1), (x2, y2), bgr, bbox_thick)

        cv2.imshow(title, img)
        cv2.waitKey(1)

    def post_processing(self, outputs):
        """
        Post-process the detection outputs. See YoloFishDetector and
        MegadetectorDetector for implementations.

        Args:
            outputs: Outputs from the detection model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")
