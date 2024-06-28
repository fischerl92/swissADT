from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import sys
from typing import Iterable
import cv2
import numpy as np
import logging
import base64


def encode_images(image_arrays: Iterable[np.ndarray]) -> Iterable[str]:
    # Convert the numpy array to bytes
    for image_array in image_arrays:
        _, buffer = cv2.imencode('.png', image_array)
        image_bytes = buffer.tobytes()
        # Encode the bytes to base64
        yield base64.b64encode(image_bytes).decode('utf-8')


def save_subclip(input_file: str, output_file: str, start_time_seconds: float, end_time_seconds: float):

    # Open the video file
     with VideoFileClip(input_file, audio=False) as video_clip:

        # Check if the end time is greater than the duration of the video
        if end_time_seconds > video_clip.duration:
            end_time_seconds = video_clip.duration

        # Check if the start time is less than 0
        if start_time_seconds < 0:
            start_time_seconds = 0

        # Extract subclip without including the timecode track
        subclip = video_clip.subclip(start_time_seconds, end_time_seconds).without_audio()

        # Write the subclip to the output file
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")


def extract_frames(video_path: str, num_frames: int = None, nth_frame: int = None) -> Iterable[np.ndarray]:
    # Open the video file
    video_clip = VideoFileClip(video_path, audio=False)

    # Get the duration of the video
    duration = video_clip.duration
    logging.info(f"Duration of the video: {duration} seconds")

    if num_frames:
        # Calculate the time intervals to extract frames, including the first and last frames
        intervals = np.linspace(0, duration, num_frames + 2)[1:-1]  # Exclude the first and last frames
        frames = (video_clip.get_frame(interval) for interval in intervals)
    elif nth_frame:
        # Calculate the time intervals to extract frames
        frames = (frame for i, frame in enumerate(video_clip.iter_frames()) if i % nth_frame == 0)

    for frame in frames:
        # Get the size of the frame in bytes
        current_image_size_bytes = sys.getsizeof(frame.tobytes())

        # Convert bytes to megabytes
        current_image_size_mb = current_image_size_bytes / (1024 ** 2)

        # Resize the frame if it's larger than 20 MB
        max_image_size_mb = 20

        if current_image_size_mb > max_image_size_mb:
            # Calculate the resizing factor
            resize_factor = (max_image_size_mb / current_image_size_mb) ** 0.5

            # Resize the frame
            resized_frame = Image.fromarray(frame)
            new_size = tuple(int(dim * resize_factor) for dim in resized_frame.size)
            resized_frame = resized_frame.resize(new_size, Image.ANTIALIAS)

            # Update the frame with the resized image
            frame = np.array(resized_frame)

        # Save the frame as a jpg image
        yield frame

    # Close the video clip
    video_clip.close()