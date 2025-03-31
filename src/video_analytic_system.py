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
            frame_index = 0
            while True:
                frame = self.video_capture.read_frame()

                if frame is None:
                    print("[INFO] Video ended.")
                    break

                if self.frame_to_skip > 0:
                    if skipped == 0:
                        skipped += 1
                    else:
                        skipped = (skipped + 1) % self.frame_to_skip
                        continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                

                # future = self.executor.submit(run_clip, self.query_list, frame)
                run_clip(self.query_list, frame, frame_index)
                frame_index += 1

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected. Cleaning up...")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
        finally:
            # self.executor.shutdown()
            self.video_capture.release()


if __name__ == "__main__":
    source = "../data/market.mp4"
    # source = "../data/road.mp4"
    
    # ✅ Ask for the Natural Language Input
    user_prompt = input("Enter what you'd like to search for in the video: ").strip() 

    # Softmax default_queries
    default_queries = ["mouse", "mug", "water bottle", "book", "orange", "computer", "gengar", "ghost", "phone", "bag"]

    # Append the user input to the first
    query_list = [user_prompt] + default_queries

    # Initialize the system
    system = VideoAnalyticSystem(source, query_list)

    # Loading Streaming Road Camera
    # system = VideoAnalyticSystem(0, ["mouse", "mug", "water bottle", "book", "apple", "computer", "gengar", "ghost", "phone", "bag"])
    system.run()