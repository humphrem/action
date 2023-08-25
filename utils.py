"""
This module provides utility functions for formatting time and percentages, and
for retrieving video file paths based on provided patterns.

Imports:
    glob: The glob module is used to retrieve files/pathnames matching a 
    specified pattern.
"""

import glob


def format_time(seconds, delimiter=":"):
    """
    Formats a given number of seconds into a string in the format mm:ss.

    Args:
        seconds (int): The number of seconds to format.
        delimiter (str, optional): The delimiter to use between minutes and
        seconds. Defaults to ":".

    Returns:
        str: The formatted time string.
    """
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}{delimiter}{int(seconds):02d}"


def format_percent(num):
    """
    Formats a given number as a percentage with 2 decimal places.

    Args:
        num (float): The number to format as a percentage.

    Returns:
        str: The formatted percentage string.
    """
    return "{:.2f}%".format(num * 100)


def get_video_paths(file_patterns):
    """
    Converts a list of file patterns into a list of matching file paths.

    Args:
        file_patterns (list): A list of file patterns to match.

    Returns:
        list: A list of file paths matching the provided patterns.
    """
    video_paths = []
    for pattern in file_patterns:
        video_paths.extend(glob.glob(pattern))
    return video_paths
