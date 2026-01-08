import os
import time
import argparse
import subprocess
import math
import numpy as np
import cv2


# Auxiliary Variates
CUTLINE = '--------------------------------' + '--------------------------------'

# Constants
POLYFIT_COEFFS_FILE_PATH = 'raw_data'
POLYFIT_COEFFS_FILE_NAME = 'polyfit_coeffs.npy'
FIT_SCRIPT = 'fit_data.py'
SAVE_MAP_NAME = 'map' # Final name is SAVE_MAP_NAME + g_args.map_model + g_args.minisize
SAVE_MAP_FOLDER = 'map_texture'

# Process Variates
g_args = None
g_center_shift_x = 0.0
g_center_shift_y = 0.0

g_tan_factors_base = 0.1
g_transition_start = 0.0
g_transition_end = 0.0


def parse_argument():
    global g_args

    parser = argparse.ArgumentParser(description='Bake Map Texture')

    parser.add_argument('-F', '--save-format', metavar='', type=str, choices=('png', 'bin'), default='png', help='saving format of texture [png(default), bin]')

    group = parser.add_argument_group('screen parameters')
    group.add_argument('-R', '--resolution', metavar=('WIDTH', 'HEIGHT'), nargs=2, type=int, default=(1920, 1080), help='screen resolution [default: 1920 1080]')

    group = parser.add_argument_group('map texture parameters')
    group.add_argument('-M', '--map-model', metavar='', type=str, choices=('keep_center', 'expand_edge', 'transition'), default='keep_center', help='map model [keep_center(default), expand_edge, transition]')
    group.add_argument('--minisize', action='store_true', help='is minisize?')
    group.add_argument('--mono', action='store_true', help='is mono map?')

    group = parser.add_argument_group('other parameters')
    group.add_argument('--swap-edge', action='store_true', help='swap edge with expand edge mode?')
    group.add_argument('-TR', '--transition-range', metavar=('START', 'END'), nargs=2, type=float, default=(0.2, 0.5), help='range in transition map model [default: 0.2 0.5]')

    g_args = parser.parse_args()

def get_polyfit_coeffs():
    file_path = POLYFIT_COEFFS_FILE_PATH + '/' + POLYFIT_COEFFS_FILE_NAME

    if os.path.isfile(file_path):
        coeffs = np.load(file_path)
        return coeffs
    else:
        script_path = POLYFIT_COEFFS_FILE_PATH + '/' + FIT_SCRIPT

        # Find none polyfit coeffs file, need to execute fit script.
        subprocess.run(['python', script_path])

    if os.path.isfile(file_path):
        coeffs = np.load(file_path)
        return coeffs
    else:
        print('Get %s error' % (file_path))
        return np.array([])

def get_r_from_pixel(x: int, y: int):
    _x = float(x) - g_center_shift_x
    _y = float(y) - g_center_shift_y
    return math.sqrt(_x*_x + _y*_y)

def get_tan_factor_based_on_keeping_center(coeffs: np.ndarray):
    poly_deriv = np.poly1d(coeffs).deriv(m=1)
    deriv_center = poly_deriv(0)

    # delta_tan_r_center / delta_real_r_center = deriv_center
    # delta_tan_r_center = delta_r_center_map / tan_factor
    # Keep center factor is 1: delta_r_center_map / delta_real_r_center = 1
    # tan_factor = 1 / deriv_center

    return 1.0 / deriv_center

def get_tan_factor_based_on_expanding_edge(coeffs: np.ndarray):
    poly = np.poly1d(coeffs)

    r_edge_x = get_r_from_pixel(0, g_args.resolution[1] / 2)
    tan_r_edge_x = poly(r_edge_x)

    r_edge_y = get_r_from_pixel(g_args.resolution[0] / 2, 0)
    tan_r_edge_y = poly(r_edge_y)

    # tan_r_edge = r_edge_map / tan_factor
    # expand edge to screen edge: r_edge_map = r_edge
    # tan_factor = r_edge / tan_r_edge

    if g_args.swap_edge:
        return min(r_edge_x / tan_r_edge_x, r_edge_y / tan_r_edge_y)

    return max(r_edge_x / tan_r_edge_x, r_edge_y / tan_r_edge_y)

def set_tan_factors(coeffs: np.ndarray):
    global g_tan_factors_base
    global g_transition_start
    global g_transition_end

    match g_args.map_model:
        case 'keep_center':
            g_tan_factors_base = get_tan_factor_based_on_keeping_center(coeffs[:, 1])

        case 'expand_edge' | 'transition':
            if g_args.mono:
                g_tan_factors_base = get_tan_factor_based_on_expanding_edge(coeffs[:, 1])
            else:
                tan_factors = np.array([
                    get_tan_factor_based_on_expanding_edge(coeffs[:, 0]),
                    get_tan_factor_based_on_expanding_edge(coeffs[:, 1]),
                    get_tan_factor_based_on_expanding_edge(coeffs[:, 2]),
                ])

                g_tan_factors_base = tan_factors.max()

        case _:
            g_tan_factors_base = np.array([0.1, 0.1, 0.1])
    
    r_max = get_r_from_pixel(0, 0)
    g_transition_start = g_args.transition_range[0] * r_max
    g_transition_end = g_args.transition_range[1] * r_max

def get_antidistortion_map_texture_based_on_minisize(coeffs: np.ndarray):
    # Only need to calculate 1/4 area by symmetry
    texture_width = g_args.resolution[0] >> 1
    texture_height = g_args.resolution[1] >> 1

    texture = np.zeros([texture_height, texture_width * 3, 4], dtype = np.uint8, order = 'C')

    # Filter 45 degree symmetry direction area
    filter_refrence = texture_width - texture_height

    for i in range(0, 3):
        if g_args.mono and i != 1:
            continue

        poly = np.poly1d(coeffs[:, i])
        x_offset = i * texture_width

        for y in range(0, texture_height):
            for x in range(0, texture_width):
                if filter_refrence >= 0:
                    if x > y + filter_refrence:
                        continue
                elif x - filter_refrence < y:
                    continue

                r_preset = get_r_from_pixel(x, y)

                if g_args.map_model == 'transition':
                    if r_preset < g_transition_start:
                        ratio = 1.0
                    else:
                        ratio = g_tan_factors_base * poly(r_preset) / r_preset

                        if r_preset < g_transition_end:
                            alpha = (r_preset - g_transition_start) / (g_transition_end - g_transition_start)
                            ratio = (1.0 - alpha) + alpha * ratio
                else:
                    ratio = g_tan_factors_base * poly(r_preset) / r_preset

                map_bytes = np.float32(ratio).view(np.uint32).tobytes()
                _x = x + x_offset
                texture[y, _x, 0] = map_bytes[0]
                texture[y, _x, 1] = map_bytes[1]
                texture[y, _x, 2] = map_bytes[2]
                texture[y, _x, 3] = map_bytes[3]

    return texture

def get_antidistortion_map_texture_based_on_quarter(coeffs: np.ndarray):
    # Only need to calculate 1/4 area by symmetry
    texture = get_antidistortion_map_texture_based_on_minisize(coeffs)
    texture_width = texture.shape[1] / 3
    texture_height = texture.shape[0]

    # Filter 45 degree symmetry direction area
    filter_refrence = texture_width - texture_height

    for i in range(0, 3):
        if g_args.mono and i != 1:
            continue

        x_offset = i * texture_width

        for y in range(0, texture_height):
            for x in range(0, texture_width):
                if filter_refrence >= 0:
                    if x > y + filter_refrence:
                        x_mirror = y - filter_refrence
                        y_mirror = x + filter_refrence
                        texture[y, x + x_offset, :] = texture[y_mirror, x_mirror + x_offset, :]
                elif x - filter_refrence < y:
                    x_mirror = y + filter_refrence
                    y_mirror = x - filter_refrence
                    texture[y, x + x_offset, :] = texture[y_mirror, x_mirror + x_offset, :]

    return texture

def get_antidistortion_map_texture(coeffs: np.ndarray):
    if g_args.minisize:
        return get_antidistortion_map_texture_based_on_minisize(coeffs)
    else:
        return get_antidistortion_map_texture_based_on_quarter(coeffs)

def save_texture(file_name: str, texture: np.ndarray):
    if not os.path.exists(SAVE_MAP_FOLDER):
        os.makedirs(SAVE_MAP_FOLDER)

    if g_args.mono:
        file_full_name = file_name + '_mono_' + g_args.map_model
        texture = texture[:, g_args.resolution[0] >> 1:g_args.resolution[0], :]
    else:
        file_full_name = file_name + '_' + g_args.map_model

    if (g_args.map_model == 'expand_edge' or g_args.map_model == 'transition') and g_args.swap_edge:
        file_full_name += '_swap_edge'

    if g_args.minisize:
        file_full_name += '_minisize'

    match g_args.save_format:
        case 'png':
            save_file_name = SAVE_MAP_FOLDER + '/' + file_full_name + '.png'
            bgra_texture = cv2.cvtColor(texture, cv2.COLOR_RGBA2BGRA) # Default color order in OpenCV is BGR
            cv2.imwrite(save_file_name, bgra_texture)
            print('Save %s success' % (save_file_name))

        case 'bin':
            save_file_name = SAVE_MAP_FOLDER + '/' + file_full_name + '.bin'
            texture.tofile(save_file_name)
            print('Save %s success' % (save_file_name))

        case _:
            print('Error format %s' % (g_args.save_format))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start Bake Antidistortion Map')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parse_argument()

    print('Map resolution: %dx%d' % (g_args.resolution[0], g_args.resolution[1]))
    print('Map model: %s' % (g_args.map_model))
    print('Minisize: %s' % (g_args.minisize))
    print('Mono: %s' % (g_args.mono))

    # Preset parameters
    g_center_shift_x = (g_args.resolution[0] - 1)*0.5
    g_center_shift_y = (g_args.resolution[1] - 1)*0.5

    polyfit_coeffs = get_polyfit_coeffs()
    if polyfit_coeffs.shape[0] > 1 and polyfit_coeffs.shape[1] == 3:
        set_tan_factors(polyfit_coeffs)

        print('Map tan factors base: %f' % (g_tan_factors_base))
        if g_args.map_model == 'transition':
            print('Transition range: [%f,%f]' % (g_transition_start, g_transition_end))

        map_texture = get_antidistortion_map_texture(polyfit_coeffs)
        if map_texture.size > 0:
            save_texture(SAVE_MAP_NAME, map_texture)
        else:
            print('Bake map texture failed')

    consume_time = time.time() - start_time
    print('End Bake Antidistortion Map')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
