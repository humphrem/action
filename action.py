import os
import glob
import time
import shutil
import logging
import argparse
import subprocess
import sys
from multiprocessing import Process, Queue, Event

import cv2
import numpy as np

# pip3 install onnxruntime-silicon
import onnxruntime


# Get the clips dir for a video
def get_clips_dir(video_path):
    return f"{os.path.splitext(video_path)[0]}_clips"


# Delete old clips directory
def remove_clips(video_path, logger):
    try:
        clips_dir = get_clips_dir(video_path)
        shutil.rmtree(clips_dir)
    except Exception as e:
        logger.warning(f"Unable to remove old clips: {e}")


# We use a converted ONNX model (using pytorch-YOLOV4's demo_darknet2onnx.py)
# of the yolov4 version YOLO-Fish https://github.com/tamim662/YOLO-Fish
def load_model():
    onnx_weights_path = "yolov4_1_3_608_608_static.onnx"
    if not os.path.exists(onnx_weights_path):
        raise FileNotFoundError(
            f"The YOLO-Fish model file {onnx_weights_path} does not exist."
        )
    session = onnxruntime.InferenceSession(
        onnx_weights_path, providers=["CPUExecutionProvider"]
    )
    return session


# We use ffmpeg to generate clips. It consumes clip requests from a queue.
def create_clip_process(queue, stop_event):
    # If the user does ctrl+c, stop this subprocess
    while not stop_event.is_set():
        try:
            clip_start_time, clip_end_time, clip_count, video_path = queue.get()

            # Check if there are any more clip requests waiting in the queue
            if clip_start_time is None:
                break

            # Create a clip for the given detection period with ffmpeg
            clip_duration = clip_end_time - clip_start_time
            clip_filename = f"{get_clips_dir(video_path)}/{(clip_count):03}-{format_time(clip_start_time, '_')}-{format_time(clip_end_time, '_')}{os.path.splitext(video_path)[1]}"
            os.makedirs(os.path.dirname(clip_filename), exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(clip_start_time),
                    "-t",
                    str(clip_duration),
                    "-i",
                    video_path,
                    clip_filename,
                ]
            )
        except KeyboardInterrupt:
            pass
        except queue.Empty:
            continue


# Add a clip request to the queue
def create_clip(clip_start_time, clip_end_time, clip_count, video_path, queue, logger):
    logger.debug(
        f"Creating clip {clip_count + 1}: start={format_time(clip_start_time)}, end={format_time(clip_end_time)}"
    )
    queue.put((clip_start_time, clip_end_time, clip_count + 1, video_path))


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
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
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(conf_thresh, nms_thresh, output):
    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    return bboxes_batch


# Define the detect function to run the yolo-fish model on a single frame of video
# and return a list of bounding boxes and confidence values for fish in the frame
def detect(session, image_src, confidence_threshold):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Format image for Input (608x608), to size and colour used by yolo-fish
    resized = cv2.resize(
        image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR
    )
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # Compute detections, run on img_in inputs, return outputs
    # turn outputs into a series of boxes with post_processing
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})
    boxes = post_processing(confidence_threshold, 0.6, outputs)
    return boxes


def plot_boxes_cv2(img, boxes):
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

        if len(box) >= 7:
            confidence = box[5]
            msg = "fish " + str(round(confidence, 3))
            text_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            top_left = (x1, y1)
            bottom_right = (top_left[0] + text_size[0], top_left[1] - text_size[1] - 10)

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
    return img


def draw_detections(image_src, boxes):
    img = plot_boxes_cv2(image_src, boxes[0])
    cv2.imshow("Video", img)
    cv2.waitKey(1)


# Format a number of seconds to mm:ss
def format_time(seconds, delimiter=":"):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}{delimiter}{int(seconds):02d}"


# Format a number to percent with 2 decimal places
def format_percent(num):
    return "{:.2f}%".format(num * 100)


# Defining the function process_frames, called in main
def process_frames(cap, session, clip_queue, fps, total_frames, logger, args):
    video_path = args.filename
    confidence_threshold = args.confidence
    buffer_seconds = args.buffer
    min_detection_duration = args.min_duration
    frames_to_skip = args.skip_frames
    show_detections = args.show_detections

    # Number of frames per minute of video time
    frames_per_minute = 60 * fps
    # Frame number for the next progress message
    next_progress_frame = frames_per_minute

    # Track when there are fish in frame as detection events
    detection_start_time = None
    detection_highest_confidence = 0
    detection_event = False

    frame_count = 0
    clip_count = 0

    # Loop over all frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        # If there isn't another frame, we're done
        if not ret:
            break

        # If a detection is already happening, then we know we're going to
        # record for a given period, so we can skip ahead to the frame where
        # the detection period ends. However, check the next frame after this
        # to see if we should end this detection period or extend it.
        if detection_event:
            skip_ahead_frames = int(min_detection_duration * fps + buffer_seconds)
            logger.debug(f"Detection event, skipping ahead {skip_ahead_frames} frames")

            # Skip ahead the number of frames that will be in the clip (i.e.,
            # don't bother checking these, they are getting recorded already)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + skip_ahead_frames)
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += skip_ahead_frames

            # Before ending this detection period, check if there is a fish
            # in what comes after it and extend if necessary
            boxes = detect(session, frame, confidence_threshold)
            if len(boxes[0]) > 0:
                detection_highest_confidence = max(
                    detection_highest_confidence, max(box[5] for box in boxes[0])
                )
                logger.info(
                    f"Fish detected, extending detection event: {format_time(frame_count / fps)} (max confidence={format_percent(detection_highest_confidence)})"
                )
                if show_detections:
                    draw_detections(frame, boxes)
                continue

            # No fish found after this clip, so we're done--create the clip
            detection_end_time = frame_count / fps + buffer_seconds
            logger.info(
                f"Detection period ended: {format_time(detection_end_time)} (duration={format_time(detection_end_time - detection_start_time)}, max confidence={format_percent(detection_highest_confidence)})"
            )
            create_clip(
                detection_start_time,
                detection_end_time,
                clip_count,
                video_path,
                clip_queue,
                logger,
            )
            clip_count += 1

            # Reset the detection period
            detection_event = False
            detection_highest_confidence = 0
            continue

        # If we're not already in a detection event, process every n frames
        # vs. every frame for speed (e.g., every 15 of 30fps)
        if frame_count % frames_to_skip == 0:
            boxes = detect(session, frame, confidence_threshold)

            # If one ore more fish were detected
            if len(boxes[0]) > 0:
                detection_highest_confidence = max(
                    detection_highest_confidence, max(box[5] for box in boxes[0])
                )

                # If we're not already in a detection event, start one
                if not detection_event:
                    detection_start_time = max(0, frame_count / fps - buffer_seconds)
                    logger.info(
                        f"Fish detected, starting detection event: {format_time(detection_start_time)} (max confidence={format_percent(detection_highest_confidence)})"
                    )
                    if show_detections:
                        draw_detections(frame, boxes)
                    detection_event = True

        # We've finished processing this frame
        frame_count += 1

        # Print a progress message every minute of video time so we know what's going on
        if frame_count >= next_progress_frame:
            logger.info(
                f"\nProgress: {format_percent(frame_count / total_frames)} processed ({frame_count}/{total_frames} frames, {format_time(frame_count / fps)})\n"
            )
            next_progress_frame += frames_per_minute

    # Before we finish the program, check if there's a detection event in progress
    # and if there is, end it now so we don't lose the final clip.
    if detection_event:
        detection_end_time = frame_count / fps + buffer_seconds
        logger.info(
            f"Detection period ended: {format_time(detection_end_time)} (duration={format_time(detection_end_time - detection_start_time)}, max confidence={format_percent(detection_highest_confidence)})"
        )
        create_clip(
            detection_start_time,
            detection_end_time,
            clip_count,
            video_path,
            clip_queue,
            logger,
        )
        clip_count += 1
    return clip_count


# Turn a list of one or more filenames, paths, globs, into a list of file paths
def get_video_paths(file_patterns):
    video_paths = []
    for pattern in file_patterns:
        video_paths.extend(glob.glob(pattern))
    return video_paths

# Main part of program to do setup and start processing frames in each file
def main(args):
    # Create a logger for this module and set the log level
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.log_level, format="%(message)s")

    video_paths = get_video_paths(args.filename)
    logger.debug(f"Got input files: {video_paths}")
    confidence_threshold = args.confidence
    buffer_seconds = args.buffer
    min_detection_duration = args.min_duration
    frames_to_skip = args.skip_frames
    delete_clips = args.delete_clips

    # Validate argument parameters from user before using them
    if buffer_seconds < 0.0:
        logger.error("Error: minimum buffer cannot be negative")
        sys.exit(1)
    
    if min_detection_duration <= 0.0:
        logger.error("Error: minimum duration must be greater than 0.0")
        sys.exit(1)
    
    if confidence_threshold <= 0.0 or confidence_threshold > 1.0:
        logger.error("Error: confidence must be greater than 0.0 and less than 1.0")
        sys.exit(1)
    
    if frames_to_skip < 0:
        logger.error("Error: frames to skip cannot be negative")
        sys.exit(1)

    cap = None
    clip_queue = None
    stop_event = Event()

    try:
        # Create a queue for clips to be processed by ffmpeg
        clip_queue = Queue()
        clip_process = Process(
            target=create_clip_process, args=(clip_queue, stop_event)
        )
        clip_process.start()

        # Loop over all the video file paths and process each one
        for i, video_path in enumerate(video_paths, start=1):
            start_time = time.time()

            # If the user requests it via -d flag, remove old clips first
            if delete_clips:
                logger.debug(f"Removing old clips from {get_clips_dir(video_path)}")
                remove_clips(video_path, logger)

            # Load YOLO-Fish model
            session = load_model()

            # Setup video capture for this video file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            logger.info(
                f"\nStarting file {i} of {len(video_paths)}: {video_path} - {format_time(duration)} - {total_frames} frames at {fps} fps, skipping every {frames_to_skip} frames"
            )
            logger.info(
                f"Using confidence threshold {confidence_threshold}, minimum clip duration of {min_detection_duration} seconds, and {buffer_seconds} seconds of buffer."
            )

            # Process the video's frames into clips of fish
            clip_count = process_frames(
                video_path, cap, session, clip_queue, fps, total_frames, logger, args
            )
            end_time = time.time()
            logger.info(
                f"Finished file {i} of {len(video_paths)}: {video_path} (total time {format_time(end_time - start_time)}). Processed {total_frames} frames into {clip_count} clips"
            )

            # Clean-up the resources we have open, if necessary
            if cap is not None:
                cap.release()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, cleaning up...")
        stop_event.set()
    except Exception as e:
        logger.error(f"There was an error: {e}")
    finally:
        # Clean-up the resources we have open, if necessary
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        # Wait for the ffmpeg clip queue to complete before we exit
        if clip_queue is not None:
            clip_queue.put((None, None, None, None))
            clip_process.join()


# Define the command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Camera Trap")
    parser.add_argument(
        "filename",
        nargs="+",  # we allow multiple file paths
        help="Path to a video file, multiple video files, or a glob pattern (e.g., ./video/*.mov)",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        default=0.0,
        type=float,
        dest="buffer",
        help="Number of seconds to add before and after detection (e.g., 1.0), cannot be negative",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        default=0.25,
        type=float,
        dest="confidence",
        help="Confidence threshold for detection (e.g., 0.45), must be greater than 0.0 and less than 1.0",
    )
    parser.add_argument(
        "-m",
        "--minimum-duration",
        default=1.0,
        type=float,
        dest="min_duration",
        help="Minimum duration for clips in seconds (e.g., 2.0), must be greater than 0.0",
    )
    parser.add_argument(
        "-f",
        "--frames-to-skip",
        default=15,
        type=int,
        dest="skip_frames",
        help="Number of frames to skip when detecting (e.g., 15), cannot be negative",
    )
    parser.add_argument(
        "-d",
        "--delete-previous-clips",
        action="store_true",
        dest="delete_clips",
        help="Whether to delete previous clips before processing video",
    )
    parser.add_argument(
        "-s",
        "--show-detections",
        action="store_true",
        dest="show_detections",
        help="Whether to show detection frames with bounding boxes",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, or ERROR)",
    )

    args = parser.parse_args()
    main(args)
