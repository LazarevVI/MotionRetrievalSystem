import os
import configparser
import ast
import time

import pandas as pd

import keypoints_extractor.extractor as ke
import keypoints_processor.preprocessor as kp
import features_computer.computer as fc
import video_loader.loader as vl

PATH_TO_CONFIG = ".//app//config//config.ini"
VIDEO_DETAILS = ".//app//video_info.csv"

config = configparser.ConfigParser()
config.read(PATH_TO_CONFIG)

VIDEO_DIRECTORY = config['Paths'].get('path_to_videos')
DATA_DIRECTORY = config['Paths'].get('path_to_df')


if __name__ == "__main__":
    if os.path.exists(VIDEO_DETAILS):
        os.remove(VIDEO_DETAILS)
    else:
        print(f"The file {VIDEO_DETAILS} does not exist")

    start = time.time()
    vl.load_yt_videos()
    end = time.time()
    print("Video loading time: ", end - start, " [seconds]")

    file_names = next(os.walk(VIDEO_DIRECTORY), (None, None, []))[2]

    for file_name in file_names:
        start = time.time()
        ke.extract(PATH_TO_CONFIG, file_name)
        end = time.time()
        print("Data extracting time: ", end - start, " [seconds]")

    start = time.time()
    kp.process_data()
    end = time.time()
    print("Data processing time: ", end - start, " [seconds]")

    video_data = pd.read_csv(VIDEO_DETAILS, sep=",",
                             header=None)

    data_file_names = next(os.walk(DATA_DIRECTORY), (None, None, []))[2]
    for idx, data_file in enumerate(data_file_names):
        data_file = DATA_DIRECTORY + data_file
        fps = int(video_data.iloc[idx, 1])
        res = ast.literal_eval(video_data.iloc[idx, 2])
        start = time.time()
        fc.compute_features(data_file, res, fps)
        end = time.time()
        print("Features computing time: ", end - start, " [seconds]")
