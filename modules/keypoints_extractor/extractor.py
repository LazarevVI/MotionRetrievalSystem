"""Keypoints extractor module"""

import os
import subprocess
import configparser
from pathlib import Path
from pymediainfo import MediaInfo
import pandas as pd


def extract(configfile: str, video_name: str):
    """Extract body keypoints from video

    Parameters
    ----------
    configfile : str
        path to configuration file
    video_name : str
        video name with extension to be processed
    """
    config = configparser.ConfigParser()
    config.read(configfile)

    cwd = os.getcwd()

    os.chdir(config["Paths"]["path_to_openpose"])
    video_path = Path(
        cwd + "\\" + config["Paths"]["path_to_videos"] + video_name)

    media_info = MediaInfo.parse(video_path)

    for track in media_info.tracks:
        if track.track_type == "Video":
            fps = track.frame_rate
            res = [track.width, track.height]

    video_data_df = pd.DataFrame(columns=['video_name', 'fps', 'res'])
    video_data_df.loc[len(video_data_df)] = [video_name, fps, res]
    video_data_df['res'] = video_data_df['res'].astype('object')
    video_data_df.to_csv(str(cwd) + "\\app\\video_info.csv",
                         mode='a+', index=False, header=False)

    keyp_path = config["Paths"]["path_to_keypoints"] + \
        os.path.splitext(video_name)[0]

    keyp_str_path = keyp_path
    keyp_path = Path(cwd + keyp_str_path)

    if not os.path.exists(keyp_path):
        os.makedirs(keyp_path)

    cmd = f"bin/OpenPoseDemo.exe --video {video_path} \
        --net_resolution {config['OpenPose']['net_resolution']} \
        --face {config['OpenPose']['face']} \
        --face_net_resolution {config['OpenPose']['face_net_resolution']} \
        --hand {config['OpenPose']['hand']} \
        --render_pose {config['OpenPose']['render_pose']} \
        --display {config['OpenPose']['display']} \
        --write_json {keyp_path}"

    subprocess.call(cmd)

    os.chdir(cwd)


if __name__ == "__main__":
    extract(".//app//config//config.ini", "video.avi")
