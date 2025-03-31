"""
Module for video stream handling and frame extraction.
Uses OpenCV to capture video frames from a live source or file.
"""

import cv2
import base64


class VideoStream:
    """
    Handles video stream input and frame extraction.
    """

    def __init__(self, source=0):
        """
        Initializes the video stream.

        Args:
            source (int or str): Video source (default is 0 for the primary webcam).
        """
        self.capture = cv2.VideoCapture(source)

    def read_frame(self):
        """
        Reads a frame from the video source.

        Returns:
            The captured frame (as a numpy array) or None if no frame is available.
        """
        ret, frame = self.capture.read()
        if not ret or frame is None or frame.size == 0:
            return None
            # raise RuntimeError("Failed to read frame")

        return frame

    def release(self):
        """
        Releases the video capture resource.
        """
        self.capture.release()
