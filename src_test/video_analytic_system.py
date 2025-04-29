import os
import torch
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future

from .video_stream import VideoStream
from .clip import run_clip, finalize_clip_session, flush_query_header
from .prompt_processor import split_joined_predicates
import time


class VideoAnalyticSystem:
    def __init__(
        self,
        video_source,
        include_queries: list[str],
        exclude_queries: list[str],
        default_queries: list[str],
        raw_query: str,
        qid: str,
        json_path: str = "output_windows.json",
    ):
        self.video_source = video_source
        self.include_queries = include_queries
        self.exclude_queries = exclude_queries
        self.default_queries = default_queries
        self.video_capture = VideoStream(self.video_source)
        self.fps = self.video_capture.fps
        self.frame_to_skip = 5
        self.executor = ThreadPoolExecutor(2)
        self.raw_query = raw_query
        self.qid = qid
        self.vid = video_source.split("/")[-1].split(".")[0]
        self.json_path = json_path

        self.avg_latency = 0.0
        self.num_frame_processed = 0

    def run(self):
        try:
            skipped = 0
            frame_index = 0
            flush_query_header(self.qid, self.raw_query, self.vid, self.fps, self.json_path)
            while True:
                frame = self.video_capture.read_frame()
                start_time = time.time()

                if frame is None:
                    print("[INFO] Video ended.")
                    break

                frame_index += 1  # ← 提前递增！

                if self.frame_to_skip > 0:
                    if skipped == 0:
                        skipped += 1
                    else:
                        skipped = (skipped + 1) % self.frame_to_skip
                        continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # future = self.executor.submit(run_clip, self.query_list, frame)
                run_clip(self.include_queries, self.exclude_queries, self.default_queries, frame, frame_index, frame_skip = self.frame_to_skip, fps=self.fps, json_path=self.json_path, raw_query=self.raw_query, qid=self.qid, vid=self.vid)
                
                latency = (time.time() - start_time) * 1000 # time spent for processing 1 frame, aka latency for 1 frame, in ms
                self.avg_latency = (self.avg_latency * self.num_frame_processed + latency) / (self.num_frame_processed + 1)
                self.num_frame_processed += 1
                # frame_index += 1
            return self.avg_latency

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected. Cleaning up...")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
        finally:
            # self.executor.shutdown()
            self.video_capture.release()
            finalize_clip_session(
                raw_query=self.raw_query,
                qid=self.qid,
                vid=self.vid,
                fps=self.fps,
                json_path="output_windows.json",
            )
            print(
                "[INFO] Completed. Saved pred_relevant_windows to output_windows.json"
            )


if __name__ == "__main__":
    source = "../data/market.mp4"
    # source = "../data/sanFrancisco.mp4"
    
    # Delete the old JSON file
    json_path = "output_windows.json"
    if os.path.exists(json_path):
        print(f"Clearing old '{json_path}'")
        os.remove(json_path)

    # Ask for the Natural Language Input
    user_prompt = input("Enter what you'd like to search for in the video: ").strip()
    include, exclude = split_joined_predicates(user_prompt)
    print(f"Include: {include}")
    print(f"Exclude: {exclude}")

    # Softmax default_queries
    default_queries = ["mouse", "mug", "water bottle", "book", "orange", "computer", "gengar", "ghost", "phone", "bag",
                       "laptop","phone","backpack","keyboard","headphones","sofa", "television","bed","lamp","plant in a pot","person sitting at a table","window with sunlight","open fridge","bookshelf","mirror"]

    # default_queries = [
    # "mouse", "mug", "water bottle", "book", "computer", "gengar", "ghost", "phone", "bag",
    # "umbrella", "cookie","table", "background", "buildings", "cars", "person",
    # "woman", "man", "standing person", "sitting person", "face", "market vendor",
    # "hand", "body", "crowd", "tree", "sky", "box", "sign", "poster", "camera"
    # ]
    
    # Initialize the system
    system = VideoAnalyticSystem(
        source,
        include,
        exclude,
        default_queries,
        raw_query=user_prompt,
    )

    # Loading Streaming Road Camera
    # system = VideoAnalyticSystem(0, default_queries)

    system.run()
