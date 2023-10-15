# Automated Camera Trapping Identification and Organization Network (ACTION)

## Overview

Action is a Python-based tool designed to bring the power of AI computer vision models to camera trap video analysis.  Using Action, it's possible to process hours of raw video into clip segments where animals or fish appear.

Whether you're monitoring aquatic life with underwater cameras or tracking terrestrial wildlife, Action can help you save time tediously scanning footage manually.

## How it Works

Action takes one or more video files as input, along with several optional parameters to customize the process, and analyzes each video.  Depending on the environment specified by the user, an appropriate object detection model is used: [YOLO-Fish v4](https://github.com/tamim662/YOLO-Fish) for aquatic videos, or [Megadetector v5](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) for terrestrial. The videos are then processed using the AI model, and whenever animals or fish are detected, a clip is created.  At the end of the process, the clips represent all the detections in the raw footage.

## Setup

Action is written in Python and requires a number of dependencies and large machine learning models (~778M) to be installed and downloaded.

The easiest way to use it is with the [pixi](https://prefix.dev/docs/pixi/overview) package manager.  Pixi installs everything you need into a local `.pixi` folder (i.e., at the root of the project), without needing to modify your system.

### Installation Steps

1. Clone this repo using `git clone https://github.com/humphrem/action.git`. NOTE: the repo includes large `.onnx` model files, which will get downloaded if you have [`git-lfs`](https://git-lfs.com/) installed (otherwise we'll get them as part of the setup below).
2. [Install pixi](https://prefix.dev/docs/pixi/overview#installation)
3. Start a terminal and navigate to the root of the Action project folder you just cloned, `cd action`
4. Enter the command `pixi run setup` to download, install, and setup everything you'll need

### Using the Pixi Shell Environment

Each time you want to use Action, you need to open a terminal and navigate to the Action folder, then start a shell with `pixi`:

```sh
pixi shell
```

This will make all of the dependencies installed with `pixi run setup` available.

When you are done, you can exit the pixi shell by using:

```sh
exit
```

## Running Action

With all dependencies installed, and the models downloaded to the `models/` directory, you can now run action:

```sh
pixi shell
python3 action.py

usage: action.py [-h] [-e {terrestrial,aquatic}] [-b BUFFER] [-c CONFIDENCE]
                 [-m MIN_DURATION] [-f SKIP_FRAMES] [-d] [-o OUTPUT_DIR] [-s]
                 [--log-level {DEBUG,INFO,WARNING,ERROR}]
                 filename [filename ...]
action.py: error: the following arguments are required: filename
```

> [!NOTE]
> On Unix systems you can also use `./action.py` without `python3`.

### Options

Action can be configured to run in different ways using various arguments and flags.

| Option | Description | Example |
| --- | --- | --- |
| `filename` | Path to a video file, multiple video files, or a glob pattern | `./video/*.mov` |
| `-e`, `--environment` | Type of camera environment, either aquatic or terrestrial. Defaults to `aquatic` | `--environment terrestrial` |
| `-b`, `--buffer` | Number of seconds to add before and after detection. Cannot be negative | `--buffer 1.0` |
| `-c`, `--confidence` | Confidence threshold for detection. Must be greater than 0.0 and less than 1.0 | `--confidence 0.45` |
| `-m`, `--minimum-duration` | Minimum duration for clips in seconds. Must be greater than 0.0 | `--minimum-duration 2.0` |
| `-f`, `--frames-to-skip` | Number of frames to skip when detecting. Cannot be negative, defaults to half the frame rate | `--frames-to-skip 15` |
| `-d`, `--delete-previous-clips` | Whether to delete previous clips before processing video | `--delete-previous-clips` |
| `-o`, `--output-dir` | Output directory to use for all clips | `--output-dir ./output` |
| `-s`, `--show-detections` | Whether to show detection frames with bounding boxes | `--show-detections` |
| `--log-level` | Logging level. Can be `DEBUG`, `INFO`, `WARNING`, or `ERROR`. Defaults to `INFO` | `--log-level DEBUG` |

> [!NOTE]
> The options with `-` or `--` are optional, while `filename` is a required argument.

### Examples

To process a video named `recording.mov` using all the default settings, specify only the filename:

```sh
python3 action.py recording.mov
```

You can also include multiple filenames:

```sh
python3 action.py recording1.mov recording2.mov recording3.mov
```

Or use a file pattern:

```sh
python3 action.py ./videos/*.avi
```

Many other options can be altered (see above) to process videos in specific ways.  For example:

```sh
python3 action.py ./video/aquatic.mov -c 0.60 -m 3.0 -s -b 1.0 -d -e aquatic
```

This would process the file `./video/aquatic.mov`, deleting all previous detections, use YOLO-Fish for detections, set a confidence threshold of `0.60` (i.e., include fish detections with confidence `0.60` and higher), make all clips `3.0` seconds minimum with a `1.0` second buffer added to the start and end of the clip (i.e., `1.0` + `3.0` + `1.0` = `5.0` seconds), and show each initial detection visual (i.e., bounding boxes on the video frame).

```sh
python3 action.py ./video/terrestrial.mov -c 0.45 -m 8.0 -b 2.0 -e terrestrial -f 25
```

This would process the file `./video/terrestrial.mov`, use Megadetector for detections, set a confidence threshold of `0.45` (i.e., include animal detections with confidence `0.45` and higher), make all clips `8.0` seconds minimum with a `2.0` second buffer added to the start and end of the clip (i.e., `2.0` + `8.0` + `2.0` = `12.0` seconds), and run detections on every 25th frame in the video.
