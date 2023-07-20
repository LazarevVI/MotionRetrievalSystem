"""Preprocessing module"""

import os
import ast
import configparser
from pathlib import Path
import json
import h5py

import pandas as pd
import numpy as np
import scipy


PATH_TO_CONFIG = './/app//config//config.ini'
config = configparser.ConfigParser()
config.read(PATH_TO_CONFIG)

PAIRS = ast.literal_eval(config['Preprocessor'].get('pairs'))
BODY_POINTS = ast.literal_eval(config['Preprocessor'].get('body_points'))
FACE_POINTS = ast.literal_eval(config['Preprocessor'].get('face_points'))
HAND_POINTS = ast.literal_eval(config['Preprocessor'].get('hand_points'))

DATA_FILE_NAME = config['Paths'].get('data_file_name')
FEATURES_FOLDER = config['Paths'].get('extracted_features')

KEY_EXTRACTED_POINTS = config['Preprocessor'].get('key_extracted_points')
KEY_PROCESSED_DATA = config['Preprocessor'].get('key_processed_data')

QX_LOW = float(config['Preprocessor'].get('qx_low'))
QX_HIGH = float(config['Preprocessor'].get('qx_high'))
QY_LOW = float(config['Preprocessor'].get('qy_low'))
QY_HIGH = float(config['Preprocessor'].get('qy_high'))
IQRX_COEFF = float(config['Preprocessor'].get('iqrx_coeff'))
IQRY_COEFF = float(config['Preprocessor'].get('iqry_coeff'))


def save_to_h5(file_name, data, name):
    path_to_file = str(file_name)
    h5f = h5py.File(path_to_file, 'a')
    h5f.create_dataset(name, data=data, dtype=float)
    h5f.close()


def delete_outliers(df):
    """
    Delete outliers in dataframe
    :param df: dataframe with outliers
    :return: dataframe without outliers
    """
    len_points = len(FACE_POINTS) + len(BODY_POINTS) + 2 * len(HAND_POINTS)

    print("Deleting outliers...")
    for part in range(len_points):

        df_part = df.loc[(slice(None), part), :]
        df_part = df_part.interpolate(method="linear")

        q1_x = df_part["x"].quantile(QX_LOW)
        q3_x = df_part["x"].quantile(QX_HIGH)
        iqr_x = q3_x - q1_x

        lower_lim_x = q1_x - IQRX_COEFF * iqr_x
        upper_lim_x = q3_x + IQRX_COEFF * iqr_x

        outliers_low_x = df_part["x"] < lower_lim_x
        outliers_up_x = df_part["x"] > upper_lim_x

        df_part["x"] = df_part["x"][~(outliers_low_x | outliers_up_x)]

        q1_y = df_part["y"].quantile(QY_LOW)
        q3_y = df_part["y"].quantile(QY_HIGH)
        iqr_y = q3_y - q1_y

        lower_lim_y = q1_y - IQRY_COEFF * iqr_y
        upper_lim_y = q3_y + IQRY_COEFF * iqr_y

        outliers_low_y = df_part["y"] < lower_lim_y
        outliers_up_y = df_part["y"] > upper_lim_y

        df_part["y"] = df_part["y"][~(outliers_low_y | outliers_up_y)]

        df_part = df_part[df_part['x' and 'y'].notna()]
        df.loc[(slice(None), part), :] = df_part

    print("Outliers deleted")

    return pd.DataFrame(df)


def interpolate_df(df: pd.DataFrame,
                   file_path: str):
    """Interpolate data in dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with data to be interpolated

    Returns
    -------
    None
    """

    len_points = len(FACE_POINTS) + len(BODY_POINTS) + 2 * len(HAND_POINTS)

    new_df_index = pd.MultiIndex.from_arrays(
        [np.repeat(np.arange(1000), len_points),
         np.tile(np.arange(len_points), 1000)],
        names=('Frame', 'Point'))

    new_df = pd.DataFrame(index=new_df_index, columns=['x', 'y'])

    print("Interpolating dataframe...")
    for part in range(len_points):
        df_part = df.loc[(slice(None), part), :]
        df_part = df_part.interpolate(method="linear")

        x = df_part['x']
        y = df_part['y']
        f = np.asarray(df_part.index.get_level_values(0))

        spl_x = scipy.interpolate.interp1d(f, x, kind="cubic")
        spl_y = scipy.interpolate.interp1d(f, y, kind="quadratic")

        fs = np.linspace(0, len(f) - 1, 1000)

        new_df.loc[(slice(None), part), "x"] = spl_x(fs)
        new_df.loc[(slice(None), part), "y"] = spl_y(fs)

    new_df = convert_to_numeric(new_df)
    new_df.to_hdf(file_path, key=KEY_PROCESSED_DATA)
    print("Interpolated dataframe saved to",
          os.path.abspath(file_path) + "\n")
    return


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe values to float

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be converted
    Returns
    -------
    pd.Dataframe
        dataframe with converted values
    """
    for column in df:
        df[column] = df[column].astype(float)
    return df


def preprocess_data(file_path: str):
    """Delete outliers and interpolate number of frames
    Parameters
    ----------
    file_path : str
        path to file with data
    Returns
    -------
    None
    """

    df = pd.DataFrame(pd.read_hdf(file_path, key=KEY_EXTRACTED_POINTS))
    df = delete_outliers(df)
    interpolate_df(df, file_path)


def get_jsons(directory) -> list:
    """Get json files names in directory

    Parameters
    ----------
    directory : str
        path to jsons

    Returns
    -------
    list
        list of json files names
    """
    jsons = [f for f in os.listdir(directory) if os.path.isfile(
        os.path.join(directory, f))]
    return jsons


def extract_data(video_data: str, json_files: list, folder_name: str) -> str:
    """Extract key data from jsons and create hdf5
    Parameters
    ----------
    video_data : str
        path to folder with json files
    json_files : list
        list of json files paths with keypoints data
    folder_name : str
        name of the folder containing json files
    Returns
    -------
    str
        Path to data file
    """

    frame = 0
    mi_keypoints = [[], []]
    keypoints_arr = np.empty((0, 2))

    cwd = os.getcwd()
    df_path = config["Paths"]["path_to_df"] + folder_name

    df_str_path = df_path
    df_path = Path(cwd + df_str_path)

    # if not os.path.exists(df_path):
    #     os.makedirs(df_path)
    #     print(f"Created new directory {df_path}")

    print("Extracting data from json files...")

    for json_file in json_files:
        with open(video_data + "//" + json_file, encoding="UTF-8") as f:

            data = json.load(f)

            df = pd.json_normalize(data["people"])

            if "pose_keypoints_2d" in df:

                pose_keypoints = df["pose_keypoints_2d"]
                face_keypoints = df["face_keypoints_2d"]
                lhand_keypoints = df["hand_left_keypoints_2d"]
                rhand_keypoints = df["hand_right_keypoints_2d"]

                pose_frame_arr = np.asarray(pose_keypoints[0])
                pose_frame_arr = np.reshape(
                    pose_frame_arr, (int(len(pose_frame_arr) / 3), 3))[:, :2]
                keypoints_arr = np.vstack((keypoints_arr, pose_frame_arr))

                face_frame_arr = np.asarray(face_keypoints[0])
                new_shape = (int(len(face_frame_arr) / 3), 3)
                face_frame_arr = np.reshape(face_frame_arr,
                                            new_shape)[FACE_POINTS, :2]
                keypoints_arr = np.vstack((keypoints_arr, face_frame_arr))

                lhand_frame_arr = np.asarray(lhand_keypoints[0])
                lhand_frame_arr = np.reshape(
                    lhand_frame_arr, (int(len(lhand_frame_arr) / 3), 3))[:, :2]
                keypoints_arr = np.vstack((keypoints_arr, lhand_frame_arr))

                rhand_frame_arr = np.asarray(rhand_keypoints[0])
                rhand_frame_arr = np.reshape(
                    rhand_frame_arr, (int(len(rhand_frame_arr) / 3), 3))[:, :2]
                keypoints_arr = np.vstack((keypoints_arr, rhand_frame_arr))

                mi_keypoints[0] = np.concatenate(
                    (mi_keypoints[0], np.full(len(FACE_POINTS) +
                                              len(BODY_POINTS) +
                                              2 * len(HAND_POINTS), frame)))
                mi_keypoints[1] = np.concatenate(
                    (mi_keypoints[1], np.arange(len(FACE_POINTS) +
                                                len(BODY_POINTS) +
                                                2 * len(HAND_POINTS))))

                frame += 1

    mi_keypoints = [list(map(int, i)) for i in mi_keypoints]
    keypoints_index = pd.MultiIndex.from_arrays(
        mi_keypoints, names=('Frame', 'Point'))
    df_keypoints = pd.DataFrame({'x': keypoints_arr[:, 0],
                                 'y': keypoints_arr[:, 1]},
                                index=keypoints_index).astype(int)
    df_keypoints.replace(0, np.nan, inplace=True)

    df_filename = str(df_path) + "_" + DATA_FILE_NAME

    df_keypoints.to_hdf(df_filename, key=KEY_EXTRACTED_POINTS, mode="w")

    print("Extracted data saved to", os.path.abspath(df_filename))

    return str(os.path.abspath(df_filename))


def process_data():
    path_to_folders_with_jsons = config["Paths"]["path_to_keypoints"]

    for folder_with_jsons in os.listdir(path_to_folders_with_jsons):
        video_data_dir = os.path.join(
            path_to_folders_with_jsons, folder_with_jsons)

        if os.path.isdir(video_data_dir):
            json_list = get_jsons(video_data_dir)

        path_to_file = extract_data(video_data_dir,
                                    json_list,
                                    folder_with_jsons)

        preprocess_data(path_to_file)


if __name__ == "__main__":
    local_json_folders = config["Paths"]["path_to_keypoints"]
    for local_folder_with_jsons in os.listdir(local_json_folders):
        local_video_data_dir = os.path.join(
            local_json_folders, local_folder_with_jsons)

        if os.path.isdir(local_video_data_dir):
            local_json_list = get_jsons(local_video_data_dir)

        local_path_to_file = extract_data(local_video_data_dir,
                                          local_json_list,
                                          local_folder_with_jsons)

        preprocess_data(local_path_to_file)
