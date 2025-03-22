import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future
from torchvision import transforms
import matplotlib.pyplot as plt

from video_stream import VideoStream
from clip import run_clip


class VideoAnalyticSystem:
    def __init__(
        self,
        video_source,
        query_list: list[str] = None,
    ):
        self.video_source = video_source
        self.query_list = query_list
        self.video_capture = VideoStream(self.video_source)
        self.frame_to_skip = 5
        self.executor = ThreadPoolExecutor(2)

    def run(self):
        try:
            skipped = 0
            while True:
                frame = self.video_capture.read_frame()

                if self.frame_to_skip > 0:
                    if skipped == 0:
                        skipped += 1
                    else:
                        skipped = (skipped + 1) % self.frame_to_skip
                        continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                for query in self.query_list:
                    future = self.executor.submit(run_clip(query, frame))
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected. Cleaning up...")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
        finally:
            self.executor.shutdown()
            self.video_capture.release()


if __name__ == "__main__":
    source = "../data/30_1742508521.mp4"
    system = VideoAnalyticSystem(0, ["mug"])
    system.run()
