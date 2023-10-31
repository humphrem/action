import os
import requests

# Get the package version from the environment variable
package_version = os.getenv("PIXI_PACKAGE_VERSION")

# URLs of the ONNX model files
model_urls = [
    f"https://github.com/humphrem/action/releases/download/v{package_version}/md_v5a_1_3_640_640_static.onnx",
    f"https://github.com/humphrem/action/releases/download/v{package_version}/yolov4_1_3_608_608_static.onnx",
]

# URLs of the sample video files
video_urls = [
    f"https://github.com/humphrem/action/releases/download/v{package_version}/aquatic_demo_1.mov",
    f"https://github.com/humphrem/action/releases/download/v{package_version}/aquatic_demo_2.mov",
    f"https://github.com/humphrem/action/releases/download/v{package_version}/terrestrial_demo_1.mov",
    f"https://github.com/humphrem/action/releases/download/v{package_version}/terrestrial_demo_2.mov",
]


# Function to download files
def download_files(urls, directory):
    # Download each file
    for url in urls:
        # Get the file name by splitting the URL and taking the last part
        file_name = url.split("/")[-1]

        # Check if the file already exists
        if not os.path.exists(os.path.join(directory, file_name)):
            print(f"Downloading {file_name} (this will take some time...)")

            # Send a HTTP request to the URL of the file
            response = requests.get(url)

            # Write the content of the response to a file in the directory
            with open(os.path.join(directory, file_name), "wb") as file:
                file.write(response.content)

            print(f"Downloaded {file_name} successfully.")
        else:
            print(f"{file_name} already exists, skipping download.")


# Create the models directory if it does not exist
if not os.path.exists("models"):
    os.makedirs("models")
    print("Created 'models' directory")

# Download model files
download_files(model_urls, "models")


# Create the video directory if it does not exist
if not os.path.exists("video"):
    os.makedirs("video")
    print("Created 'video' directory")

# Download video files
download_files(video_urls, "video")
