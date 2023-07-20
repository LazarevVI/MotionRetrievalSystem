"""Module for computing movement features"""

import os
import ast
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from scipy.interpolate import griddata
import keypoints_processor.preprocessor as kp

VIDEO_DETAILS = ".//app//video_info.csv"

PATH_TO_CONFIG = './/app//config//config.ini'
config = configparser.ConfigParser()
config.read(PATH_TO_CONFIG)

FEATURES_DIRECTORY = config['Paths'].get('extracted_features')

KEY_PROCESSED_DATA = config['Preprocessor'].get('key_processed_data')
KEY_FEATURES = config['Preprocessor'].get('key_processed_data')
KEY_ROT = config['Preprocessor'].get('key_rot')
KEY_DIV = config['Preprocessor'].get('key_div')
KEY_PLANE = config['Preprocessor'].get('key_plane')

WIN_WE = int(config['Preprocessor'].get('win_we'))
WIN_SE = int(config['Preprocessor'].get('win_se'))
WIN_PE = int(config['Preprocessor'].get('win_pe'))
POINTS_VF = ast.literal_eval(config['Preprocessor'].get('points_vf'))

PAIRS = ast.literal_eval(config['Preprocessor'].get('pairs'))
BODY_POINTS = ast.literal_eval(config['Preprocessor'].get('body_points'))
FACE_POINTS = ast.literal_eval(config['Preprocessor'].get('face_points'))
HAND_POINTS = ast.literal_eval(config['Preprocessor'].get('hand_points'))
WEIGHTS = ast.literal_eval(config['Preprocessor'].get('weights'))


def compute_features(data_file_path: str, res: list, fps: int):

    data = pd.DataFrame(pd.read_hdf(data_file_path, key=KEY_PROCESSED_DATA))

    data = center_of_mass(data, PAIRS, WEIGHTS)
    data = speed(data, fps)
    data = acceleration(data, fps)
    data = weight_effort(data, WIN_WE)
    data = space_effort(data, WIN_SE)

    file_path = Path(FEATURES_DIRECTORY +
                     os.path.splitext(os.path.basename(data_file_path))[0] +
                     "_features.hdf5")

    data = kp.convert_to_numeric(data)
    data.to_hdf(file_path, key=KEY_FEATURES)

    data = pd.DataFrame(pd.read_hdf(file_path),
                        columns=['x', 'y', 'speed', 'v_x', 'v_y',
                                 'acceleration', 'weight', 'space'])
    _, _, rot, diverge = velocity_field(data, POINTS_VF, res)

    kp.save_to_h5(file_path, rot, KEY_ROT)
    kp.save_to_h5(file_path, diverge, KEY_DIV)

    plane_feature = plane(data, WIN_PE)
    kp.save_to_h5(file_path, plane_feature, KEY_PLANE)

    return


def get_seg_center_coords(df, pairs):
    """
    Get coords of body segment centers in each frame
    :param pairs: pairs of points that form a body segment
    :param df: coordinates of keypoints in each frame
    :return: seg_centers - coords of body segment centers in each frame
    """

    seg_centers = np.zeros(
        (len(df.index.unique(level=0)), int(len(pairs) / 2), 2))

    print("Computing segment centers...")
    for j in range(0, len(pairs), 2):
        part_1 = pairs[j]
        part_2 = pairs[j + 1]

        part_1_coord = df.loc[(slice(None), part_1), ["x", "y"]]
        part_2_coord = df.loc[(slice(None), part_2), ["x", "y"]]

        center_x = np.mean(np.asarray(
            [part_1_coord["x"], part_2_coord["x"]]), axis=0)
        center_y = np.mean(np.asarray(
            [part_1_coord["y"], part_2_coord["y"]]), axis=0)

        seg_centers[:, int(j / 2), 0] = center_x
        seg_centers[:, int(j / 2), 1] = center_y
    print("Done\n")
    return seg_centers


def center_of_mass(df, pairs, w):
    """
    Get coords of body center of mass in each frame
    :param df: dataframe with body keypoints coordinates
    :param pairs: pairs of points that form a body segment
    :param w: weight contribution for each segment
    :return: dataframe with added coords of body center of mass
    """

    frames_number = len(df.index.unique(level=0))
    keyp_number = len(df.index.unique(level=1))
    seg_centers = get_seg_center_coords(df, pairs)
    cm_coords = np.zeros((frames_number, 2))

    print('Computing center of mass coordinates...')
    for i in range(frames_number):
        cm_coords[i][0] = np.average(seg_centers[i, :, 0], weights=w)
        cm_coords[i][1] = np.average(seg_centers[i, :, 1], weights=w)

    cm_index = pd.MultiIndex.from_arrays([np.arange(frames_number),
                                          np.full(frames_number, keyp_number)],
                                         names=('Frame', 'Point'))
    cm_df = pd.DataFrame(cm_coords, columns=["x", "y"], index=cm_index)

    new_df = pd.concat([df, cm_df]).sort_index()

    print("Done\n")

    return new_df


def speed(df, fps):
    """
    :param df: coordinates of keypoints in each frame
    :param fps: frames per second for video
    :return: dataframe with speed of keypoints
                        in each frame and v_x, v_y components
    """
    print('Computing speed...')
    df["speed"] = (df.groupby("Point")["x"].diff().pow(2) +
                   df.groupby("Point")["y"].diff().pow(2)).pow(0.5).mul(fps)
    df["speed"] = df["speed"].groupby("Point").shift(-1)
    df["v_x"] = df.groupby("Point")["x"].diff() * fps
    df["v_x"] = df["v_x"].groupby("Point").shift(-1)
    df["v_y"] = df.groupby("Point")["y"].diff() * fps
    df["v_y"] = df["v_y"].groupby("Point").shift(-1)
    print("Done\n")

    return df


def acceleration(df, fps):
    """
    Acceleration of every body point in each frame
    :param df: parameters of keypoints in each frame
    :param fps: frames per second for video
    :return: acceleration of keypoints in each frame
    """
    print('Computing acceleration...')
    df["acceleration"] = df.groupby("Point")["speed"].diff() * fps
    df["acceleration"] = df["acceleration"].groupby("Point").shift(-1)
    print("Done\n")
    return df


def weight_effort(df, win):
    """
    Weight effort parameter [Laban movement analysis]
                                for every body point per frame_interval
    :param df: parameters of keypoints in each frame
    :param win: rolling window size
    :return: dataframe with weight effort parameter
    """
    print('Computing weight effort...')
    df["weight"] = df.groupby("Point")["speed"].rolling(
        win).mean().droplevel(0).shift(int(-win / 2))
    print("Done\n")
    return df


def space_effort(df, win):
    """
    Space effort parameter [Laban movement analysis]
                                for every body point per frame_interval
    :param df: parameters of keypoints in each frame
    :param win: rolling window size
    :return: dataframe with space effort parameter
    """
    print('Computing space effort...')
    df["space"] = (df.groupby("Point")["x"].diff() ** 2 +
                   df.groupby("Point")["y"].diff() ** 2) \
        .groupby("Point").rolling(win) \
        .sum() \
        .droplevel(0) / \
        (df.groupby("Point")["x"].diff(periods=win) ** 2 +
         df.groupby("Point")["y"].diff(periods=win) ** 2)
    df["space"] = df["space"]**0.5
    print("Done\n")
    return df


def plane(df, win):
    """
    Plane effort parameter [Laban movement analysis]
                                for every body point per frame_interval
    :param win: rolling window size
    :param df: parameters of keypoints in each frame
    :return: dataframe with plane parameters
    """
    print('Computing plane effort...')
    keyp_num = len(df.index.unique(level=1))
    new_df = pd.DataFrame()
    new_df["plane_x"] = (df.loc[idx[:, :(keyp_num-2)], ["x"]] -
                         df.xs(keyp_num-1, level=1)[["x"]]).groupby("Frame")\
        .sum()\
        .div(keyp_num-1)\
        .rolling(win)\
        .sum()
    new_df["plane_y"] = (df.loc[idx[:, :(keyp_num-2)], ["y"]] -
                         df.xs(keyp_num-1, level=1)[["y"]]).groupby("Frame")\
        .sum()\
        .div(keyp_num-1)\
        .rolling(win)\
        .sum()
    print("Done\n")
    new_df.dropna()
    return new_df


def interpolate_velocity_field(df, points=0, res=(1280, 720)):
    """
    Interpolate vector field for each pixel of video
    :param df: parameters of keypoints in each frame
    :param points: which keypoints to interpolate
    :param res: video resolution
    :return: frames with interpolated velocity components
    """

    _df = df.loc[(slice(None), points), ["x", "y", "v_x", "v_y"]
                 ].groupby(['x', 'y']).mean()

    x0 = _df.index.get_level_values(0)
    y0 = _df.index.get_level_values(1)
    u0 = _df.loc[:, "v_x"]
    v0 = _df.loc[:, "v_y"]

    end1 = complex(0, res[0])
    end2 = complex(0, res[1])

    grid_x, grid_y = np.mgrid[0:(res[0] - 1):end1, 0:(res[1]-1):end2]
    coords = np.vstack((x0, y0)).T

    grid_u0 = griddata(coords, u0, (grid_x, grid_y), method='linear')
    grid_v0 = griddata(coords, v0, (grid_x, grid_y), method='linear')

    return grid_u0, grid_v0


def velocity_field(df, points=0, res=(1280, 720)):

    print('Computing velocity field...')
    u0, v0 = interpolate_velocity_field(df, points, res)
    u0, v0 = crop_vector_field(u0, v0)
    print("Done\n")

    vect_x = u0
    vect_y = v0
    diverge = divergence(u0, v0)
    rot = curl(u0, v0)

    return vect_x, vect_y, rot, diverge


def crop_vector_field(_u0, _v0):
    """Compute divergence for velocity vector field of keypoints

    :param _u0: grid of x velocity components
    :type _u0: np.ndarray
    :param _v0: grid of y velocity components
    :type _v0: np.ndarray
    :param dpx: delta x and delta y for central derivative computing,
                defaults to 1
    :type dpx: int, optional
    :return: divergence for velocity vector field
    :rtype: np.ndarray
    """
    u0_not_nan_ind = np.argwhere(~np.isnan(_u0))
    v0_not_nan_ind = np.argwhere(~np.isnan(_v0))

    not_nan_ind = np.vstack((u0_not_nan_ind[[0, -1]], v0_not_nan_ind[[0, -1]]))

    not_nan_beg = not_nan_ind[[0, -1]].min(axis=0)
    not_nan_end = not_nan_ind[[0, -1]].max(axis=0)

    u0_cropped = _u0[not_nan_beg[0]:not_nan_end[0],
                     not_nan_beg[1]:not_nan_end[1]]
    v0_cropped = _u0[not_nan_beg[0]:not_nan_end[0],
                     not_nan_beg[1]:not_nan_end[1]]

    return u0_cropped, v0_cropped


def divergence(_u0, _v0, dpx=1):
    """Compute divergence for velocity vector field of keypoints

    :param _u0: grid of x velocity components
    :type _u0: np.ndarray
    :param _v0: grid of y velocity components
    :type _v0: np.ndarray
    :param dpx: delta x and delta y for central derivative computing,
                defaults to 1
    :type dpx: int, optional
    :return: divergence for velocity vector field
    :rtype: np.ndarray
    """

    diverge = np.zeros(_u0.shape)
    print("Computing divergence...")
    for _idx, _ in np.ndenumerate(diverge):
        if _idx[0] + dpx < _u0.shape[0] and _idx[1] + dpx < _u0.shape[1]:
            div_x = (_u0[_idx[0] + dpx, _idx[1]] -
                     _u0[_idx[0] - dpx, _idx[1]]) / (2 * dpx)
            div_y = (_v0[_idx[0], _idx[1] + dpx] -
                     _v0[_idx[0], _idx[1] - dpx]) / (2 * dpx)
            diverge[_idx] = div_y + div_x
    print("Done\n")
    return diverge


def curl(_u0: np.ndarray, _v0: np.ndarray, dpx: int = 1) -> np.ndarray:
    """Compute curl for velocity vector field of keypoints

    :param _u0: grid of x velocity components
    :type _u0: np.ndarray
    :param _v0: grid of y velocity components
    :type _v0: np.ndarray
    :param dpx: delta x and delta y for central derivative computing,
                defaults to 1
    :type dpx: int, optional
    :return: curl for velocity vector field
    :rtype: np.ndarray
    """
    rot = np.zeros(_u0.shape)
    print("Computing curl...")
    for _idx, _ in np.ndenumerate(rot):
        if _idx[0] + dpx < _u0.shape[0] and _idx[1] + dpx < _u0.shape[1]:
            cx = (_v0[_idx[0] + dpx, _idx[1]] -
                  _v0[_idx[0] - dpx, _idx[1]]) / (2 * dpx)
            cy = (_u0[_idx[0], _idx[1] + dpx] -
                  _u0[_idx[0], _idx[1] - dpx]) / (2 * dpx)
            rot[_idx] = cx - cy
    print("Done\n")
    return rot
