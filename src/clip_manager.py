"""
This module is responsible for managing the creation of video clips. As
a detector finds objects of interest, video clips are extracted. The majority
of this work is done by the ClipManager class, with support from some other
utility functions.
"""

import os
import shutil
import subprocess
import platform
from multiprocessing import Process, Queue, Event
from queue import Empty

from .utils import format_time

import cv2


def get_clips_dir(video_path):
    """
    Get the clips directory for a video path. We use the
    video's basename + _clips (e.g., movie -> movie_clips/).

    Args:
        video_path (str): The path to the video file.

    Returns:
        str: The path to the clips directory.
    """
    return f"{os.path.splitext(video_path)[0]}_clips"


def remove_clips_dir(video_path, logger):
    """
    Delete the old clips directory for the given video_path,

    Args:
        video_path (str): The path to the video file.
        logger (Logger): The logger object.

    Returns:
        None
    """
    clips_dir = get_clips_dir(video_path)
    remove_output_dir(clips_dir, logger)


def remove_output_dir(output_dir, logger):
    """
    Delete a common output dir for video clips.

    Args:
        output_dir (str): The path to remove.
        logger (Logger): The logger object.

    Returns:
        None
    """
    try:
        logger.debug(f"Removing old clips from {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Unable to remove old clips: {e}")


def create_output_dir(output_dir):
    """
    Make sure the output dir exists

    Args:
        output_dir (str): The path to create.

    Returns:
        None
    """
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"{output_dir} is a file, not a directory.")
    else:
        os.makedirs(output_dir, exist_ok=True)


class ClipManager:
    """
    A class used to manage the creation of video clips. Video clips
    are created using ffmpeg (must be installed on the system)
    in a separate process.

    Attributes:
        logger (Logger): The logger object.
        output_dir (string): Optional base path for storing clips
        clip_queue (Queue): The queue of clips.
        stop_event (Event): The stop event for the clip process.
        clip_process (Process): The clip process.
        clip_count (int): The current clip count.
        bbox_count (int): The current bbox image count.
    """

    def __init__(self, logger, output_dir):
        """
        The constructor for the ClipManager class. It manages
        the queue and process for ffmpeg.

        Args:
            logger (Logger): The logger object.
            output_dir (string): Optional base path for storing clips
        """
        self.logger = logger
        self.output_dir = output_dir
        self.clip_queue = Queue()
        self.stop_event = Event()

        # Get the best codecs for this system
        self.decoder, self.encoder = self.get_ffmpeg_codecs()
        self.logger.info(f"Using decoder={self.decoder}, encoder={self.encoder}")

        self.clip_process = Process(
            target=self.create_clip_process,
            args=(self.clip_queue, self.stop_event, self.decoder, self.encoder),
        )
        self.clip_process.start()

        self.clip_count = 0
        self.bbox_count = 0

    def get_ffmpeg_codecs(self):
        """Return tuple of (decoder, encoder) codecs"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Prefer hardware accelerated encoding on MacOS
            if platform.system() == "Darwin" and "h264_videotoolbox" in result.stdout:
                return ("h264", "h264_videotoolbox")  # Use h264 decoder with hw encoder
            return ("h264", "libx264")  # software fallback
        except subprocess.SubprocessError:
            return ("h264", "libx264")

    def create_clip_process(self, queue, stop_event, decoder, encoder):
        """
        Create a clip process using ffmpeg. It consumes clip requests from a queue.

        Args:
            queue (Queue): The queue of clips.
            stop_event (Event): The stop event for the clip process.

        Returns:
            None
        """
        # Create an MP4 video clip from the overall video
        # but, stop this subprocess if the user does ctrl+c
        while not stop_event.is_set():
            try:
                clip_start_time, clip_end_time, clip_count, video_path = queue.get()

                # Check if there are any more clip requests waiting in the queue
                if clip_start_time is None:
                    break

                # Create a clip for the given detection period with ffmpeg
                clip_duration = clip_end_time - clip_start_time
                base_dir = (
                    self.output_dir if self.output_dir else get_clips_dir(video_path)
                )
                clip_filename = f"{base_dir}/{(clip_count):04}-{format_time(clip_start_time, '_')}-{format_time(clip_end_time, '_')}.mp4"
                create_output_dir(os.path.dirname(clip_filename))

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
                        "-c:v",
                        decoder,
                        "-i",
                        video_path,
                        "-c:v",
                        encoder,
                        "-c:a",
                        "aac",  # use Advanced Audio Coding (AAC) for audio compatibility
                        "-pix_fmt",  # use YUV planar color space with 4:2:0 chroma subsampling (QuickTime)
                        "yuv420p",  # see https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
                        clip_filename,
                    ]
                )
            except KeyboardInterrupt:
                pass
            except Empty:
                continue

    def create_clip(self, clip_start_time, clip_end_time, video_path):
        """
        Add a clip request to the queue, which will eventually
        create a new clip file.

        Args:
            clip_start_time (float): The start time of the clip.
            clip_end_time (float): The end time of the clip.
            video_path (str): The path to the video file.

        Returns:
            None
        """
        self.clip_count += 1
        self.logger.debug(
            f"Creating clip {self.clip_count}: start={format_time(clip_start_time)}, end={format_time(clip_end_time)}"
        )
        self.clip_queue.put(
            (clip_start_time, clip_end_time, self.clip_count, video_path)
        )

    def create_bbox_image(self, clip_time, bbox_img, video_path):
        """
        Write a bounding box image to the clips directory

        Args:
            clip_time (float): The time of the bounding box.
            bbox_img: The bounding box image.
            video_path (str): The path to the video file.

        Returns:
            None
        """

        self.bbox_count += 1

        # Create a bbox image for the given detection
        base_dir = self.output_dir if self.output_dir else get_clips_dir(video_path)
        bbox_filename = (
            f"{base_dir}/{(self.bbox_count):04}-{format_time(clip_time, '_')}.jpg"
        )
        create_output_dir(os.path.dirname(bbox_filename))

        # Write the bbox image to the clips directory as a JPG
        cv2.imwrite(bbox_filename, bbox_img)

    def stop(self):
        """
        Let the queue know it's time to stop processing new clip
        requests so we can quit.

        Returns:
            None
        """
        self.stop_event.set()

    def cleanup(self):
        """
        Wait for the ffmpeg clip queue to complete whatever job(s)
        are happening now, before we exit.

        Returns:
            None
        """
        if self.clip_queue is not None:
            self.logger.debug("Waiting for remaining clips to be saved")
            self.clip_queue.put((None, None, None, None))
            self.clip_process.join()

    def reset(self):
        """
        Reset the clip and bbox counts to 0.

        Returns:
            None
        """
        self.logger.debug("Resetting clip manager counts to 0")
        self.clip_count = 0
        self.bbox_count = 0

    def get_clip_count(self):
        """
        Get the current clip count.

        Returns:
            int: The current clip count.
        """
        return self.clip_count
