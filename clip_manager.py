"""
This module is responsible for managing the creation of video clips. As
a detector finds objects of interest, video clips are extracted. The majority
of this work is done by the ClipManager class, with support from some other
utility functions.
"""


import os
import shutil
import subprocess
from multiprocessing import Process, Queue, Event
from queue import Empty

from utils import format_time


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
    if exists.

    Args:
        video_path (str): The path to the video file.
        logger (Logger): The logger object.

    Returns:
        None
    """
    try:
        clips_dir = get_clips_dir(video_path)
        logger.debug(f"Removing old clips from {clips_dir}")
        shutil.rmtree(clips_dir)
    except Exception as e:
        logger.warning(f"Unable to remove old clips: {e}")


class ClipManager:
    """
    A class used to manage the creation of video clips. Video clips
    are created using ffmpeg (must be installed on the system)
    in a separate process.

    Attributes:
        logger (Logger): The logger object.
        clip_queue (Queue): The queue of clips.
        stop_event (Event): The stop event for the clip process.
        clip_process (Process): The clip process.
    """

    def __init__(self, logger):
        """
        The constructor for the ClipManager class. It manages
        the queue and process for ffmpeg.

        Args:
            logger (Logger): The logger object.
        """
        self.logger = logger
        self.clip_queue = Queue()
        self.stop_event = Event()
        self.clip_process = Process(
            target=self.create_clip_process, args=(self.clip_queue, self.stop_event)
        )
        self.clip_process.start()

    def create_clip_process(self, queue, stop_event):
        """
        Create a clip process using ffmpeg. It consumes clip requests from a queue.

        Args:
            queue (Queue): The queue of clips.
            stop_event (Event): The stop event for the clip process.

        Returns:
            None
        """
        # Create a clip from the overall video
        # but, stop this subprocess if the user does ctrl+c
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
            except Empty:
                continue

    def create_clip(self, clip_start_time, clip_end_time, clip_count, video_path):
        """
        Add a clip request to the queue, which will eventually
        create a new clip file.

        Args:
            clip_start_time (float): The start time of the clip.
            clip_end_time (float): The end time of the clip.
            clip_count (int): The count of the clip.
            video_path (str): The path to the video file.

        Returns:
            None
        """
        self.logger.debug(
            f"Creating clip {clip_count}: start={format_time(clip_start_time)}, end={format_time(clip_end_time)}"
        )
        self.clip_queue.put((clip_start_time, clip_end_time, clip_count, video_path))

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
