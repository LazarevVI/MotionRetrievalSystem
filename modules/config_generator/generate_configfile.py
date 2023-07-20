"""Deafault config generator"""

import configparser
import numpy as np

# pairs of connected body keypoints
PAIRS = [1, 8,
         2, 3,
         3, 4,
         5, 6,
         6, 7,
         9, 10,
         10, 11,
         12, 13,
         13, 14,
         1, 0]

WEIGHTS = [0.33020, 0.03075,  # weight fraction of body-segments
           0.02295, 0.03075,
           0.02295, 0.11125,
           0.06430, 0.11125,
           0.06430, 0.06810]

# face keypoints to be extracted
FACE_POINTS = [2, 8, 14, 17, 19, 21, 22, 24, 26, 48, 51, 54, 57, 68, 69]

# hand keypoints
HAND_POINTS = list(np.arange(21))

# body keypoints
BODY_POINTS = list(np.arange(25))

# window size
WIN_SE = 60
WIN_WE = 60
WIN_PE = 60


# resolution
RES = (1280, 720)

# points for velocity field
POINTS_VF = [0, 1, 2]

#
QX_LOW = 0.1
QX_HIGH = 0.9
QY_LOW = 0.1
QY_HIGH = 0.9

IQRX_COEFF = 0.85
IQRY_COEFF = 0.85

KEY_EXTRACTED_POINTS = "RAW_DATA"
KEY_PROCESSED_DATA = "PROCESSED_DATA"
KEY_FEATURES = "FEATURES"
KEY_ROT = "ROTOR"
KEY_DIV = "DIVERGENCE"
KEY_PLANE = "PLANE"


def generate():
    """Generate config file"""

    path_to_config = './/app//config//config.ini'
    config = configparser.ConfigParser()
    config.read(path_to_config)

    config['Paths'] = {'home': './/app//',
                       'path_to_config': './/app//config//config.ini',
                       'path_to_openpose': './/dependencies//openpose//',
                       'path_to_videos': './/app//data//videos//',
                       'path_to_keypoints': './/app//data//keypoints//',
                       'extracted_features': './/app//data//features//',
                       'path_to_df': './/app//data//dataframes//',
                       'data_file_name': 'data.h5',
                       }

    config['OpenPose'] = {'net_resolution': "-1x304",
                          'face_net_resolution': "320x320",
                          'render_pose': '0',
                          'display': '0',
                          'hand': 'true',
                          'face': 'true'
                          }

    config['Preprocessor'] = {
        'pairs': PAIRS,
        'weights': WEIGHTS,
        'face_points': FACE_POINTS,
        'hand_points': HAND_POINTS,
        'body_points': BODY_POINTS,
        'win_se': WIN_SE,
        'win_we': WIN_WE,
        'win_pe': WIN_PE,
        'points_vf': POINTS_VF,
        'key_extracted_points': KEY_EXTRACTED_POINTS,
        'key_processed_data': KEY_PROCESSED_DATA,
        'key_features': KEY_FEATURES,
        'key_rot': KEY_ROT,
        'key_div': KEY_DIV,
        'key_plane': KEY_PLANE,

    }

    with open(path_to_config, 'w', encoding="UTF-8") as configfile:
        config.write(configfile)


if __name__ == "__main__":
    generate()
