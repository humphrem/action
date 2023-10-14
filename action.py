"""
This is the main module for ACTION. It handles parsing arguments from the user,
loading and managing resources, and processing detections into clips.
"""

import os
import sys
import time
import logging
import argparse

from clip_manager import (
    ClipManager,
    remove_clips_dir,
    remove_output_dir,
    create_output_dir,
)
from yolo_fish_detector import YoloFishDetector
from megadetector_detector import MegadetectorDetector
from utils import format_time, format_percent, get_video_paths

import cv2


# We use converted ONNX models for YOLO-Fish (https://github.com/tamim662/YOLO-Fish)
# and Megadetector (https://github.com/microsoft/CameraTraps)
def load_detector(environment, min_duration, buffer, confidence, logger):
    """
    Load the appropriate detector based on the environment provided.

    Args:
        environment (str): The type of detector to load. Must be either "aquatic" or "terrestrial".
        min_duration (float): The minimum duration of a generated clip
        buffer (float): An optional number of seconds to add before/after a clip
        confidence (float): The confidence level to use
        logger (logging.Logger): The logger to use for logging messages.

    Returns:
        detector (object): An instance of the appropriate detector.

    Raises:
        TypeError: If the any args are not of the correct type
    """

    # Make sure any user-provided flags for detection are valid before we use them
    if buffer and buffer < 0.0:
        raise TypeError("Error: minimum buffer cannot be negative")

    if min_duration and min_duration <= 0.0:
        raise TypeError("Error: minimum duration must be greater than 0.0")

    if confidence and (confidence <= 0.0 or confidence > 1.0):
        raise TypeError("Error: confidence must be greater than 0.0 and less than 1.0")

    detector = None
    if environment == "terrestrial":
        detector = MegadetectorDetector(logger, min_duration, buffer, confidence)
    elif environment == "aquatic":
        detector = YoloFishDetector(logger, min_duration, buffer, confidence)
    else:
        raise TypeError("environment must be one of aquatic or terrestrial")

    detector.load()
    return detector


# Defining the function process_frames, called in main
def process_frames(
    video_path,
    cap,
    detector,
    clips,
    fps,
    total_frames,
    frames_to_skip,
    logger,
    args,
):
    """
    Process frames from a video file and create clips based on detections.

    Args:
        video_path (str): The path to the video file.
        cap (cv2.VideoCapture): The video capture object.
        detector (object): The detector to use for detecting objects in frames.
        clips (ClipManager): The clip manager for managing clips.
        fps (int): The frames per second of the video.
        total_frames (int): The total number of frames in the video.
        frames_to_skip (int): The number of frames to skip between detections
        logger (logging.Logger): The logger to use for logging messages.
        args (argparse.Namespace): The command line arguments.

    Returns:
        None
    """
    buffer_seconds = detector.buffer
    min_detection_duration = detector.min_duration
    show_detections = args.show_detections

    # Number of frames per minute of video time
    frames_per_minute = 60 * fps
    # Frame number for the next progress message
    next_progress_frame = frames_per_minute

    # Track when there are is something in frame as detection events
    detection_start_time = None
    detection_highest_confidence = 0
    detection_event = False

    frame_count = 0

    # Loop over all frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        # If there isn't another frame, we're done
        if not ret:
            break

        # If a detection is already happening, then we know we're going to
        # record for a given period, so we can skip ahead to the frame where
        # the detection period ends. However, we then need to check *all* the
        # frames after this within the buffer period (i.e., so we don't overlap
        # with the end buffer period in the previous clip).  If we detect something
        # in these frames, we should extend this detection period; otherwise
        # we end it and create a new clip.
        if detection_event:
            # Calculate the number of frames to skip ahead. We know that we
            # want to record for a minimum duration, and potentially we add
            # a buffer period in seconds.
            skip_ahead_frames = int((min_detection_duration + buffer_seconds) * fps)
            logger.debug(f"Detection event, skipping ahead {skip_ahead_frames} frames")

            # Skip ahead the number of frames that will be in the clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + skip_ahead_frames)
            frame_count += skip_ahead_frames

            # Check some frames within the buffer period for a detection, to see if
            # we should extend this detection period, or if it's OK to end it. NOTE:
            # if `buffer_seconds` is 0, at least check the next frame.
            for i in range(max(1, int(buffer_seconds * fps))):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Only process frames that are multiples of frames_to_skip
                # or the last frame in the video.
                if i % frames_to_skip == 0 or frame_count == total_frames - 1:
                    logger.debug(
                        f"Checking frame {frame_count} before ending detection event"
                    )
                    # If a detection is made, extend the current detection period
                    boxes = detector.detect(frame)
                    if len(boxes) > 0:
                        detection_highest_confidence = max(
                            detection_highest_confidence, max(box[4] for box in boxes)
                        )
                        logger.info(
                            f"{detector.class_name} detected, extending detection event: {format_time(frame_count / fps + buffer_seconds)} (max confidence={format_percent(detection_highest_confidence)})"
                        )
                        if show_detections:
                            detector.draw_detections(frame, boxes, video_path)
                        break
            else:
                # If no detection was made within the buffer period, and we didn't
                # extend, end the detection period now and create a new clip.
                detection_end_time = frame_count / fps + buffer_seconds
                logger.info(
                    f"Detection period ended: {format_time(detection_end_time)} (duration={format_time(detection_end_time - detection_start_time)}, max confidence={format_percent(detection_highest_confidence)})"
                )
                clips.create_clip(
                    detection_start_time,
                    detection_end_time,
                    video_path,
                )

                # Reset the detection period
                detection_event = False
                detection_highest_confidence = 0

                # Start again reading the next frame
                continue

        # If we're not already in a detection event, process every n frames
        # vs. every frame for speed (e.g., every 15 of 30fps). We also check
        # the last frame, so we don't miss anything at the edge.
        if frame_count % frames_to_skip == 0 or frame_count == total_frames - 1:
            boxes = detector.detect(frame)

            # If there are one ore more detections
            if len(boxes) > 0:
                detection_highest_confidence = max(
                    detection_highest_confidence, max(box[4] for box in boxes)
                )

                # If we're not already in a detection event, start one
                if not detection_event:
                    detection_start_time = max(0, frame_count / fps - buffer_seconds)
                    logger.info(
                        f"{detector.class_name} detected, starting detection event: {format_time(frame_count / fps)} (max confidence={format_percent(detection_highest_confidence)})"
                    )
                    if show_detections:
                        detector.draw_detections(frame, boxes, video_path)
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
        clips.create_clip(
            detection_start_time,
            detection_end_time,
            video_path,
        )


# Main part of program to do setup and start processing frames in each file
def main(args):
    """
    The main function of the program. Sets up the logger, validates arguments,
    loads the detector, and processes frames from each video file.

    Args:
        args (argparse.Namespace): The command line arguments.
    """

    # Create a logger for this module and set the log level
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.log_level, format="%(message)s")

    delete_clips = args.delete_clips
    output_dir = args.output_dir
    video_paths = get_video_paths(args.filename)
    logger.debug(f"Got input files: {video_paths}")

    # Validate argument parameters from user before using them
    if len(video_paths) < 1:
        logger.error("Error: you must specify one or more video filenames to process")
        sys.exit(1)

    # Load YOLO-Fish or Megadetector, based on `-e` value
    detector = None
    try:
        detector = load_detector(
            args.environment, args.min_duration, args.buffer, args.confidence, logger
        )
    except Exception as e:
        logger.error(f"There was an error: {e}")
        sys.exit(1)

    cap = None
    clips = None

    # Initialize the output_dir if specified
    if output_dir:
        # If `-d`` was specified, delete old clips first
        if delete_clips:
            remove_output_dir(output_dir, logger)
        # Create the output directory if it doesn't exist
        create_output_dir(output_dir)

    try:
        # Create a queue manager for clips to be processed by ffmpeg
        clips = ClipManager(logger, output_dir)

        # Keep track of total time to process all files, recording start time
        total_time_start = time.time()

        # Loop over all the video file paths and process each one
        for i, video_path in enumerate(video_paths, start=1):
            # Make sure this video path actually exists before we try to use it
            if not os.path.exists(video_path):
                logger.info(f"Video path {video_path} does not exist, skipping.")
                continue

            file_start_time = time.time()

            # If the user requests it via -d flag, and isn't using a common output_dir
            # remove old clips first
            if not output_dir and delete_clips:
                remove_clips_dir(video_path, logger)

            # Setup video capture for this video file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_to_skip = args.skip_frames or int(fps / 2.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            logger.info(
                f"\nStarting file {i} of {len(video_paths)}: {video_path} - {format_time(duration)} - {total_frames} frames at {fps} fps, skipping every {frames_to_skip} frames"
            )
            logger.info(
                f"Using confidence threshold {detector.confidence}, minimum clip duration of {detector.min_duration} seconds, and {detector.buffer} seconds of buffer."
            )

            # If we're not using a common clips dir, reset the counter for future clips
            if not output_dir:
                clips.reset_clip_count()

            clip_count_before = clips.get_clip_count()

            # Process the video's frames into clips
            process_frames(
                video_path,
                cap,
                detector,
                clips,
                fps,
                total_frames,
                frames_to_skip,
                logger,
                args,
            )

            clip_count_after = clips.get_clip_count()
            clips_processed = clip_count_after - clip_count_before

            file_end_time = time.time()
            logger.info(
                f"Finished file {i} of {len(video_paths)}: {video_path} (total time to process file {format_time(file_end_time - file_start_time)}). Processed {total_frames} frames into {clips_processed} clips"
            )

            # Clean-up the resources we have open, if necessary
            if cap is not None:
                cap.release()

            cv2.destroyAllWindows()
            cv2.waitKey(1)

        # Keep track of total time to process all files, recording end time
        total_time_end = time.time()
        logger.info(
            f"\nFinished. Total time for {len(video_paths)} files: {format_time(total_time_end - total_time_start)}"
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, cleaning up...")
        clips.stop()
    except Exception as e:
        logger.error(f"There was an error: {e}")
    finally:
        # Clean-up the resources we have open, if necessary
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Wait for the ffmpeg clip queue to complete before we exit
        if clips is not None:
            clips.cleanup()


# Define the command line arguments
if __name__ == "__main__":
    """
    The entry point of the program. Defines the command line arguments and
    calls the main function.
    """
    parser = argparse.ArgumentParser(
        description="Automated Camera Trapping Identification and Organization Network (ACTION)"
    )
    parser.add_argument(
        "filename",
        nargs="+",  # we allow multiple file paths
        help="Path to a video file, multiple video files, or a glob pattern (e.g., ./video/*.mov)",
    )
    parser.add_argument(
        "-e",
        "--environment",
        choices=["terrestrial", "aquatic"],
        default="aquatic",
        dest="environment",
        help="Type of camera environment, either aquatic or terrestrial, defaults to --environment aquatic",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        default=None,
        type=float,
        dest="buffer",
        help="Number of seconds to add before and after detection (e.g., 1.0), cannot be negative",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=None,
        dest="confidence",
        help="Confidence threshold for detection (e.g., 0.45), must be greater than 0.0 and less than 1.0",
    )
    parser.add_argument(
        "-m",
        "--minimum-duration",
        type=float,
        default=None,
        dest="min_duration",
        help="Minimum duration for clips in seconds (e.g., 2.0), must be greater than 0.0",
    )
    parser.add_argument(
        "-f",
        "--frames-to-skip",
        type=int,
        dest="skip_frames",
        help="Number of frames to skip when detecting (e.g., 15), cannot be negative, defaults to half the frame rate",
    )
    parser.add_argument(
        "-d",
        "--delete-previous-clips",
        action="store_true",
        dest="delete_clips",
        help="Whether to delete previous clips before processing video",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory to use for all clips",
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
