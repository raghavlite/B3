from datasets import load_dataset
import pandas as pd


ds = load_dataset("lmms-lab/YouCook2", split="test")
youcook = pd.read_csv("archive/validation_youcook.csv")
youcook_set = set(zip(youcook["video_id"], youcook["text"]))
unique_video_ids = set(youcook["video_id"].unique())

all_id = set()
tt = 0
for row in ds:
    video_id = row["youtube_id"]
    if video_id in unique_video_ids:
        breakpoint()
    # text = row["sentence"]
    # if (video_id, text) in youcook_set:
    #     tt += 1

breakpoint()
