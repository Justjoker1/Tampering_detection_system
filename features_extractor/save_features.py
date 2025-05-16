import features_extractor
from test import *
import os
import numpy as np
import pandas as pd
dataset_path = r"../dataset_split"
three = ["train", "val", "test"]

for one in three:
    video_path = os.path.join(dataset_path, one, "video")
    video_list = os.listdir(video_path)
    out_path = os.path.join(dataset_path, one, "stagetwo")
    os.makedirs(out_path, exist_ok=True)
    out_path1 = os.path.join(out_path, "features")
    out_path2 = os.path.join(out_path, "events")

    os.makedirs(out_path1, exist_ok=True)
    os.makedirs(out_path2, exist_ok=True)
    for video in video_list:
        path = os.path.join(video_path, video)

        # 提取不带后缀的文件名（关键修改点）
        filename_with_ext = os.path.basename(video)  # 原始带后缀的文件名（如：ori82.mp4）
        filename = os.path.splitext(filename_with_ext)[0]  # 去除后缀，得到：ori82

        baseframe, event_counts_df = features_extractor.extract_features(
            video_path=path,
            result_path="/",
            save_video=False,
            show_plot=False,
            show_text=False
        )

        csv_file_path = os.path.join(out_path1, filename + ".csv")
        csv_file_path_event = os.path.join(out_path2, filename + "_event.csv")

        baseframe.to_csv(csv_file_path, index=False)
        event_counts_df.to_csv(csv_file_path_event, index=False)
        print(f"DataFrame已成功写入 {csv_file_path}")