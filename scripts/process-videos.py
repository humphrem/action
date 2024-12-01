#!/usr/bin/env python3

import argparse
import fcntl
import subprocess
import platform
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Tuple
import shutil
import atexit
import json
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

"""
Video Processing and Merging Tool
=================================

This script processes and merges video files using ffmpeg, creating two output videos
from a series of input videos. Processing can be interrupted and resumed at any time.

Output Videos:
-------------
1. {prefix}-set.MOV:
   - 5-minute clip starting at specified time from the first video
   - Used to capture initial settling period of experiment

2. {prefix}-bat.MOV:
   - Remainder of first video (after settling period)
   - Concatenated with all subsequent videos
   - Maintains video quality and synchronization

Usage:
------

After running `pixi shell`:

1. Initial Processing:
   python process-videos.py -s START_TIME -p PREFIX [-b BATCH_SIZE] video1.MOV [video2.MOV video3.MOV ...]

2. Resume Previous Run:
   python process-videos.py
   (Will detect and offer to resume the most recent interrupted processing)

Arguments:
----------
Required:
    videos           : One or more input video files
    -s, --start-time : Time to start settling period, format MM:SS (e.g., "1:15")

Optional:
    -p, --prefix     : Prefix for output filenames (default: "video")
    -b, --batch-size : Number of videos to process simultaneously (default: 2)

Examples:
---------
Start new processing:
    python process-videos.py -s 1:15 -p experiment1 video1.MOV video2.MOV video3.MOV

Resume most recent processing:
    python process-videos.py

Features:
---------
- Interruptible: Can be stopped and resumed at any point
- Batch Processing: Process multiple videos simultaneously
- Auto-Resume: Running without arguments resumes the last interrupted run
- Progress Tracking: Shows completion status and remaining work
- State Preservation: Maintains processing state between runs
- Output Protection: Checks for existing files before overwriting
"""


class ProcessingStage(Enum):
    """Enum to track processing stages with ordering"""

    NOT_STARTED = 0
    SETTLING_CREATED = 1
    SEGMENTS_PREPARED = 2
    MERGED = 3
    COMPLETED = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class SegmentInfo:
    """Track information about each segment"""

    input_file: str
    output_file: str
    start_time: Optional[str]
    completed: bool = False


class ProcessingState:
    """ProcessingState to track where we are in the process, so we can resume"""

    def __init__(self):
        self.stage: ProcessingStage = ProcessingStage.NOT_STARTED
        self.segments: Dict[str, SegmentInfo] = {}  # key: output_file
        self.settling_output: Optional[str] = None
        self.merged_output: Optional[str] = None
        self.command_args: Optional[Dict] = None
        self.videos: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage.value,
            "segments": {
                k: {
                    "input_file": v.input_file,
                    "output_file": v.output_file,
                    "start_time": v.start_time,
                    "completed": v.completed,
                }
                for k, v in self.segments.items()
            },
            "settling_output": self.settling_output,
            "merged_output": self.merged_output,
            "command_args": self.command_args,
            "videos": self.videos,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProcessingState":
        state = cls()
        state.stage = ProcessingStage(data["stage"])
        state.segments = {
            k: SegmentInfo(
                input_file=v["input_file"],
                output_file=v["output_file"],
                start_time=v["start_time"],
                completed=v["completed"],
            )
            for k, v in data["segments"].items()
        }
        state.settling_output = data["settling_output"]
        state.merged_output = data["merged_output"]
        state.command_args = data.get("command_args")
        state.videos = data.get("videos")
        return state


class VideoProcessor:
    TEMP_ROOT = os.path.expanduser("~/.action")

    def __init__(self, prefix: str):
        """Initialize with a temporary working directory"""
        self.temp_dir = os.path.join(self.TEMP_ROOT, f"{prefix}_processing_temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        # See if we have in-progress state
        self.state_file = os.path.join(self.temp_dir, "processing_state.json")
        self.state = self._load_state()
        # Register cleanup on program exit
        atexit.register(self.cleanup_temp_dir)
        # Get decoder and encoder codecs
        self.decoder, self.encoder = self.get_ffmpeg_codecs()
        print(f"INFO: using decoder={self.decoder}, encoder={self.encoder}")

    @classmethod
    def get_last_run(cls) -> Optional[Tuple[str, ProcessingState]]:
        """Get the most recent processing directory and state"""
        if not os.path.exists(cls.TEMP_ROOT):
            return None

        temp_dirs = [
            d for d in os.listdir(cls.TEMP_ROOT) if d.endswith("_processing_temp")
        ]
        if not temp_dirs:
            return None

        # Get most recently modified temp dir
        last_dir = max(
            temp_dirs, key=lambda d: os.path.getmtime(os.path.join(cls.TEMP_ROOT, d))
        )
        state_file = os.path.join(cls.TEMP_ROOT, last_dir, "processing_state.json")

        if os.path.exists(state_file):
            with open(state_file) as f:
                state = ProcessingState.from_dict(json.load(f))
                return last_dir, state
        return None

    def _load_state(self) -> ProcessingState:
        """Load processing state from file or create new state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                # Use previous state
                return ProcessingState.from_dict(json.load(f))
        # Otherwise, we're just starting
        return ProcessingState()

    def _save_state(self):
        """Save current processing state to file"""
        with open(self.state_file, "w") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(self.state.to_dict(), f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_output_directory(self, first_video: str) -> str:
        """Get the directory path of the first video for output files"""
        return os.path.dirname(os.path.abspath(first_video))

    def check_output_permissions(self, directory: str) -> None:
        """Check if we have write permissions in the output directory"""
        if not os.access(directory, os.W_OK):
            raise PermissionError(
                f"No write permission in output directory: {directory}"
            )

    def get_temp_path(self, filename: str) -> str:
        """Get full path for a temporary file"""
        return os.path.join(self.temp_dir, filename)

    def cleanup_temp_dir(self):
        """Remove temporary directory with partial file removal and resume functionality"""
        if self.state.stage == ProcessingStage.COMPLETED:
            if os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    print("Cleaned up temporary files")
                except Exception as e:
                    print(f"Warning: Could not clean up temp files: {e}")
        else:
            # Remove any partial/failed segment files
            for segment in self.state.segments.values():
                if not segment.completed and os.path.exists(segment.output_file):
                    try:
                        os.remove(segment.output_file)
                    except OSError:
                        pass
            print("Keeping temporary files for potential resume")

    def time_to_seconds(self, time_str: str) -> int:
        """Convert time string (MM:SS) to seconds"""
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def seconds_to_time(self, seconds: int) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_video_duration(self, filename: str) -> float:
        """Get duration of video in seconds using ffprobe"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def run_ffmpeg_with_output(self, cmd: List[str], prefix: str = "") -> None:
        """Run ffmpeg command with real-time output handling"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        def handle_output(stream, prefix):
            for line in stream:
                # Skip empty lines
                if line.strip():
                    # Add prefix to each line for identification
                    print(f"{prefix}{line}", end="", flush=True)

        # Create threads to handle stdout and stderr
        from threading import Thread

        stdout_thread = Thread(
            target=handle_output, args=(process.stdout, f"[{prefix}] ")
        )
        stderr_thread = Thread(
            target=handle_output, args=(process.stderr, f"[{prefix}] ")
        )

        # Start threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete
        process.wait()

        # Wait for output threads to complete
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    def create_ffmpeg_command(
        self,
        input_file: str,
        output_file: str,
        start_time: str = None,
        duration: int = None,
    ) -> List[str]:
        """Create standardized ffmpeg command with optimal encoding settings"""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"]

        if start_time is not None:
            cmd.extend(["-ss", start_time])

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(
            [
                "-c:v",
                self.decoder,
                "-i",
                input_file,
                # Encoder
                "-c:v",
                self.encoder,
                "-c:a",
                "aac",  # AAC audio codec for compatibility
                "-pix_fmt",
                "yuv420p",  # YUV420p color space for QuickTime compatibility
                output_file,
            ]
        )

        return cmd

    def create_settling_period_video(
        self, input_file: str, output_file: str, start_seconds: int, duration: int = 300
    ) -> None:
        """Create the settling period video segment"""
        if self.state.stage >= ProcessingStage.SETTLING_CREATED:
            print(
                f"✓ Settling period video already created, {output_file}, skipping..."
            )
            return

        print("Creating settling period video...")

        # Update state before starting the process
        self.state.settling_output = output_file
        self._save_state()

        cmd = self.create_ffmpeg_command(
            input_file=input_file,
            output_file=output_file,
            start_time=self.seconds_to_time(start_seconds),
            duration=duration,
        )

        try:
            self.run_ffmpeg_with_output(cmd, "setting")
            self.state.stage = ProcessingStage.SETTLING_CREATED
            self._save_state()
            print(f"✓ Finished creating settling period video: {output_file}")
        except Exception:
            # Save state even on failure
            self._save_state()
            raise

    def process_video_segment(
        self,
        input_file: str,
        output_file: str,
        start_time: str = None,
        prefix: str = "",
    ) -> str:
        """Process a single video segment"""
        try:
            cmd = self.create_ffmpeg_command(
                input_file=input_file, output_file=output_file, start_time=start_time
            )
            self.run_ffmpeg_with_output(cmd, prefix)
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error processing video segment: {input_file}")
            print(f"ffmpeg output: {e.stdout}\n{e.stderr}")
            raise

    def prepare_video_segments(
        self,
        videos: List[str],
        start_seconds: int,
        settling_duration: int,
        batch_size: int = 2,
    ) -> List[str]:
        """Prepare all video segments for merging in parallel"""
        if self.state.stage >= ProcessingStage.SEGMENTS_PREPARED:
            completed = [s.output_file for s in self.state.segments.values()]
            print("✓ All segments already prepared, skipping...")
            print(f"  Found {len(completed)} completed segments")
            return completed

        print(f"Creating video segments for merging with batch size={batch_size}...")

        # Initialize segment tracking if not already done
        if not self.state.segments:
            # Handle first video specially (if needed)
            first_duration = self.get_video_duration(videos[0])
            if start_seconds + settling_duration < first_duration:
                first_temp = self.get_temp_path("trimmed_first.MOV")
                self.state.segments[first_temp] = SegmentInfo(
                    input_file=videos[0],
                    output_file=first_temp,
                    start_time=self.seconds_to_time(start_seconds + settling_duration),
                    completed=False,
                )

            # Set up remaining videos
            for i, video in enumerate(videos[1:], 1):
                temp_file = self.get_temp_path(f"temp_segment_{i}.MOV")
                self.state.segments[temp_file] = SegmentInfo(
                    input_file=video,
                    output_file=temp_file,
                    start_time=None,
                    completed=False,
                )
            self._save_state()

        # Get the total number of segments
        total_segments = len(self.state.segments)

        # Get incomplete segments
        incomplete_segments = [
            segment for segment in self.state.segments.values() if not segment.completed
        ]

        # Track how many segments we've completed before this batch
        previously_completed = total_segments - len(incomplete_segments)
        completed = 0

        # Process in batches
        for i in range(0, len(incomplete_segments), batch_size):
            batch = incomplete_segments[i : i + batch_size]
            completed += len(batch)
            print(
                f"Processing batch {(i // batch_size) + 1} "
                f"({completed}/{total_segments} segments)..."
            )
            print("=" * 80)  # Add separator between batches

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for j, segment in enumerate(batch):
                    segment_num = previously_completed + i + j + 1
                    segment_prefix = f"{segment_num}/{total_segments} {os.path.basename(segment.input_file)}"
                    futures.append(
                        executor.submit(
                            self.process_video_segment,
                            segment.input_file,
                            segment.output_file,
                            segment.start_time,
                            segment_prefix,
                        )
                    )

                # Wait for batch to complete
                for future, segment in zip(futures, batch):
                    try:
                        output_file = future.result()
                        print(f"Completed segment: {output_file}")
                        # Update state
                        self.state.segments[output_file].completed = True
                        self._save_state()
                    except Exception as e:
                        print(f"Error processing {segment.input_file}: {e}")
                        raise

            completed += len(batch)
            print("\n" + "=" * 80)
            time.sleep(1)

        # All segments completed
        self.state.stage = ProcessingStage.SEGMENTS_PREPARED
        self._save_state()

        # Return all segment files in correct order
        return [s.output_file for s in self.state.segments.values()]

    def get_ffmpeg_codecs(self):
        """Return tuple of (decoder, encoder) codecs"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Prefer hardware accelerated on MacOS
            if platform.system() == "Darwin" and "h264_videotoolbox" in result.stdout:
                return ("h264", "h264_videotoolbox")
            return ("h264", "libx264")  # software fallback
        except subprocess.SubprocessError:
            return ("h264", "libx264")

    def merge_video_segments(self, segment_files: List[str], output_file: str) -> None:
        """Merge all video segments into final output"""
        if self.state.stage >= ProcessingStage.MERGED:
            print(f"✓ Videos already merged to {output_file}, skipping...")
            return

        print(f"Merging {len(segment_files)} segments...")

        # Create concat file with absolute paths
        concat_file = self.get_temp_path("concat_list.txt")
        with open(concat_file, "w") as f:
            for segment in segment_files:
                # Use absolute paths and escape special characters
                abs_path = os.path.abspath(segment)
                f.write(f"file '{abs_path}'\n")

        # Add small delay before merging
        time.sleep(1)

        try:
            # First attempt with concat demuxer
            merge_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "info",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-c:v",
                self.decoder,
                "-i",
                concat_file,
                "-c:v",
                self.encoder,
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                output_file,
            ]

            try:
                self.run_ffmpeg_with_output(merge_cmd, "merge")
            except subprocess.CalledProcessError:
                print("First merge attempt failed, trying alternative method...")

                # Create intermediate list for complex filter
                filter_parts = []
                for i in range(len(segment_files)):
                    filter_parts.append(f"[{i}:v][{i}:a]")

                filter_complex = (
                    "".join(filter_parts)
                    + f"concat=n={len(segment_files)}:v=1:a=1[outv][outa]"
                )

                input_parts = []
                for segment in segment_files:
                    input_parts.extend(["-i", segment])

                alternative_cmd = (
                    ["ffmpeg", "-hide_banner", "-loglevel", "info"]
                    + input_parts
                    + [
                        "-y",
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        "[outv]",
                        "-map",
                        "[outa]",
                        "-c:v",
                        self.encoder,
                        "-c:a",
                        "aac",
                        "-pix_fmt",
                        "yuv420p",
                        output_file,
                    ]
                )

                self.run_ffmpeg_with_output(alternative_cmd, "merge")

            self.state.merged_output = output_file
            self.state.stage = ProcessingStage.MERGED
            self._save_state()

        except subprocess.CalledProcessError as e:
            print(f"Error during video merge: {e}")
            raise
        finally:
            # Ensure concat file is removed
            if os.path.exists(concat_file):
                os.remove(concat_file)


def check_system_resources():
    """Check system resources and ffmpeg availability"""
    try:
        # Check ffmpeg version
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg is not available")

        # Check available file handles
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 1024:
            # Try to increase the limit if possible
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
            except ValueError:
                print("Warning: Limited file handles available")
    except Exception as e:
        print(f"System check failed: {e}")
        raise


def cleanup_system_resources():
    """Force cleanup of system resources"""
    import gc

    gc.collect()
    time.sleep(1)


def parse_arguments(allow_empty: bool = False) -> argparse.Namespace:
    """Parse command line arguments"""

    # Add validation for time format
    def validate_time(time_str):
        try:
            if not time_str.count(":") == 1:
                raise ValueError
            mins, secs = map(int, time_str.split(":"))
            if not (0 <= secs < 60):
                raise ValueError
            return time_str
        except ValueError as e:
            raise argparse.ArgumentTypeError("Time must be in MM:SS format") from e

    # Add validation for batch size
    def validate_batch_size(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("Batch size must be at least 1")
        return ivalue

    parser = argparse.ArgumentParser(description="Process and merge video files.")

    if not allow_empty:
        parser.add_argument("videos", nargs="+", help="List of input video files")
        parser.add_argument(
            "-s",
            "--start-time",
            type=validate_time,
            required=True,
            help="Start time in MM:SS format (e.g., 1:15)",
        )
    else:
        parser.add_argument("videos", nargs="*", help="List of input video files")
        parser.add_argument(
            "-s",
            "--start-time",
            type=validate_time,
            required=False,
            help="Start time in MM:SS format (e.g., 1:15)",
        )

        parser.add_argument(
            "-p",
            "--prefix",
            default="video",
            help="Prefix for output filenames (default: video)",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=validate_batch_size,
            default=2,
            help="Number of videos to process at once (default: 2)",
        )

    return parser.parse_args()


def main():
    args = parse_arguments(allow_empty=True)

    # If no arguments provided, try to resume last run
    if len(sys.argv) == 1:
        last_run = VideoProcessor.get_last_run()
        if last_run is None:
            print("No previous run found to resume. Please provide command arguments.")
            return

        temp_dir, state = last_run
        if state.command_args is None or state.videos is None:
            print("Previous run state is incomplete. Please provide command arguments.")
            return

        print("\n=== Resuming Last Run ===")
        print(f"Command: {' '.join(sys.argv[0:1] + state.command_args)}")
        print(f"Videos: {', '.join(state.videos)}")
        response = input("Continue this processing? (Y/n): ")
        if response.lower() == "n":
            return

        # Reconstruct arguments
        sys.argv = [sys.argv[0]] + state.command_args + state.videos
        args = parse_arguments()
    elif not args.videos or not args.start_time:
        # If not resuming and missing required args, show error
        parser = argparse.ArgumentParser()
        parser.print_help()
        return

    if len(args.videos) < 1:
        print("Error: At least one video file is required")
        return

    # Check if input files exist
    for video in args.videos:
        if not os.path.exists(video):
            print(f"Error: Input file not found: {video}")
            return

    check_system_resources()

    processor = VideoProcessor(args.prefix)
    if processor.state.stage == ProcessingStage.NOT_STARTED:
        # Store command arguments in state
        processor.state.command_args = sys.argv[1 : -len(args.videos)]
        processor.state.videos = args.videos
        processor._save_state()

    # If we're resuming, check if we need to continue
    if processor.state.stage != ProcessingStage.NOT_STARTED:
        print(
            f"Resuming from stage: {processor.state.stage.name} ({processor.state.stage.value})"
        )

    # Convert start time to seconds
    start_seconds = processor.time_to_seconds(args.start_time)
    settling_duration = 300  # 5 minutes in seconds

    # Get output directory from first video
    output_dir = processor.get_output_directory(args.videos[0])
    processor.check_output_permissions(output_dir)

    # Define output filenames using prefix and output directory
    settling_output = os.path.join(output_dir, f"{args.prefix}-set.MOV")
    merged_output = os.path.join(output_dir, f"{args.prefix}-bat.MOV")

    # Check if output files already exist
    for output_file in [settling_output, merged_output]:
        if (
            os.path.exists(output_file)
            and processor.state.stage == ProcessingStage.NOT_STARTED
        ):
            response = input(
                f"Output file {output_file} already exists. Overwrite? (y/N): "
            )
            if response.lower() != "y":
                print("Aborting.")
                return

    try:
        processor.create_settling_period_video(
            args.videos[0], settling_output, start_seconds, settling_duration
        )
        cleanup_system_resources()

        temp_segments = processor.prepare_video_segments(
            args.videos, start_seconds, settling_duration, args.batch_size
        )
        cleanup_system_resources()

        processor.merge_video_segments(temp_segments, merged_output)

        # Mark as completed only if everything succeeds
        processor.state.stage = ProcessingStage.COMPLETED
        processor._save_state()

        print(f"✓ Finished merging into merged video: {merged_output}")

    except KeyboardInterrupt:
        print(
            "\nProcess interrupted. Progress has been saved and can be resumed by re-running same command."
        )
        # Ensure state is saved before exiting
        processor._save_state()
        sys.exit(1)
    except Exception as e:
        # Also save state on other exceptions
        processor._save_state()
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
