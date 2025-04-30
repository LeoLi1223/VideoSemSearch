# CS2270 Project: Natural Language Queries for Video Data Flows

## Introduction
This is the codebase of our CLIP-based real-time video semantic analysis system, which supports free-form natural language queries for streaming video input. This README provides instructions on how to set up and run our system.

## Prerequisites & Setup
1. After cloning this repository and switching into the directory where it is located, install the project dependencies by running:
```
pip install -r requirements.txt
```
2. You will also need to have a `OPENAI_API_KEY` before runninng our model. After you obtained your API key from OpenAI platform, in your terminal, run:
```
export OPENAI_API_KEY=YOUR_API_KEY
```
3. To run evaluation, you might need to download QVHighlights raw video data from [here](https://github.com/jayleicn/moment_detr/tree/main/data).


## Run Inference On A Video File
1. Switch into the `src/` directory:
```
cd src
```
2. Open `video_analytic_system.py`. Then on line 86, 
```
source = "../data/market.mp4" # CHANGE TO YOUR VIDEO SOURCE!
```
Change `../data/market.mp4` to the path where your desired video file is located.

3. In `clip.py`, on line 13-14:
```
softmax_threshold = 0.995 # For Inclusion Query
exclude_threshold = 0.20 # For Exclusion Query
```
These are the thresholds we provided for you. You can also adjust them on a 0~1 scale if you like.

4. In your terminal, run:
```
python3 video_analytic_system.py
```
5. After your program starts, you will see a prompt in your terminal:
```
Enter what you'd like to search for in the video: 
```
Please enter your full query sentence here and press Enter.

6. Run the program and wait until it finishes. Captured video frames will be stored under the `matched_images/` folder under your current working directory.

## Run Inference On Webcam Feed
1. Switch into the `src/` directory:
```
cd src
```
2. Open `video_analytic_system.py`. Then on line 86, 
```
source = "../data/market.mp4" # CHANGE TO YOUR VIDEO SOURCE!
```
Change this line to:
```
source = 0
```
To connect to your computer's webcam feed.

3. In `clip.py`, on line 13-14:
```
softmax_threshold = 0.995 # For Inclusion Query
exclude_threshold = 0.20 # For Exclusion Query
```
These are the thresholds we provided for you. You can also adjust them on a 0~1 scale if you like.

4. In your terminal, run:
```
python3 video_analytic_system.py
```
5. After your program starts, you will see a prompt in your terminal:
```
Enter what you'd like to search for in the video: 
```
Please enter your full query sentence here and press Enter.

6. Run the program and when you are ready, press `Ctrl+C` to stop program execution. Captured video frames will be stored under the `matched_images/` folder under your current working directory.


## Run Evaluation Dataset
1. If you want to modify the softmax thresholds for evaluation, go to `config/config.py` and modify them on a 0~1 scale.

2. Switch into the `test/` directory:
```
cd test
```

3. Open `run_test.py`. On line 33 and line 48, it has:
```
video_source = f"../data/{test_data_json['vid']}.mp4"
```
Change `../data` to where your QVHighlights raw videos folder is located.

4. On line 307, it has:
```
run_val_data(50, args)
```
The first argument `50` indicates the number of query-video pairs to be evaluated is 50. You can change this number according to your needs.

5. In your terminal, if you would like to evaluate the predicate-split version, run:
```
python3 run_test.py --split
```
Otherwise, if you would like to evaluate the non-split version, run:
```
python3 run_test.py
```
Wait until the program finishes to get your evaluation result. The result files should be seen under your current working directory.

