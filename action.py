#!/usr/bin/env python3

import argparse
from src.action import main

if __name__ == "__main__":
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
        "-i",
        "--include-bbox-images",
        action="store_true",
        dest="include_bbox_images",
        help="Whether to include the bounding box images for the frames that trigger or extend each detection event, along with the videos in the clips directory.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, or ERROR)",
    )

    args = parser.parse_args()
    main(args)
