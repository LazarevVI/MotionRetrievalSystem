"""Youtube video loader"""

import os
from pytube import YouTube

VIDEO_SAVE_DIRECTORY = ".//app//data//videos"
AUDIO_SAVE_DIRECTORY = ".//app//data//audios"
YT_URLS = ".//app//yt_urls.txt"
CWD = str(os.getcwd())


def download_video(video_url, idx):
    """Download video from youtube

    Parameters
    ----------
    video_url : str
        link to video
    idx : int
        video index for naming
    """

    try:
        yt = YouTube(video_url)
        video = yt.streams.filter(
            file_extension='mp4').get_highest_resolution()
        video.download(CWD + VIDEO_SAVE_DIRECTORY, filename=idx+".mp4")

    except Exception as e:
        print(f"Error '{e}' occured.")
        print("Failed to download video")

    print("Video was downloaded successfully")


def download_audio(video_url, idx):
    """Download audio from video in youtube

    Parameters
    ----------
    video_url : str
        link to video
    idx : int
        video index for naming
    """

    try:
        video = YouTube(video_url)
        audio = video.streams.filter(only_audio=True).first()
        audio.download(CWD + AUDIO_SAVE_DIRECTORY, filename=idx+".mp3")

    except Exception as e:
        print(f"Error '{e}' occured")
        print("Failed to download audio")

    print("Audio was downloaded successfully")


def read_yt_urls(path_to_txt_file: str):
    """Read lines from txt file that contains urls to youtube videos

    Parameters
    ----------
    path_to_txt_file : str
        path to txt file with urls

    Returns
    -------
    list
        list of urls to videos
    """
    with open(path_to_txt_file, encoding="UTF-8") as f:
        lines = f.readlines()
        urls_list = []

        for line in lines:
            urls_list.append(line.strip())

    return urls_list


def load_yt_videos():
    """Load videos from youtube
    """
    urls = read_yt_urls(YT_URLS)
    for idx, url in enumerate(urls):
        idx = str(idx)
        download_video(url, idx)
        download_audio(url, idx)


if __name__ == "__main__":
    load_yt_videos()
