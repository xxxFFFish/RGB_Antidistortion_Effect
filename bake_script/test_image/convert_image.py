import os
import time
import argparse
import subprocess
import cv2
import numpy as np


# Auxiliary Variates
CUTLINE = '--------------------------------' + '--------------------------------' + '--------------------------------'

# Constants
MAP_TEXTURE_FOLDER = '../map_texture'
MAP_TEXTURE_NAME = 'map'
BAKE_MAP_SCRIPT_FOLDER = '..'
BAKE_MAP_SCRIPT = 'bake_map.py'
SOURCE_IMAGE_FOLDER = 'input'
SAVE_IMAGE_FOLDER = 'output'

VALID_IMAGE_FORMAT = {'.png', '.jpg', '.jpeg', '.bmp'}

# Process Variates
g_args = None
g_width = 0
g_height = 0
g_map_shift = 0


def parse_argument():
    global g_args

    parser = argparse.ArgumentParser(description='Translate Pattern Image By Antidistortion map texture')
    parser.add_argument('--mono', action='store_true', help='is mono color antidistortion?')
    parser.add_argument('-M', '--map-model', metavar='', type=str, choices=('keep_center', 'expand_edge', 'transition'), default='keep_center', help='map model [keep_center(default), expand_edge, transition]')
    parser.add_argument('-S', '--split', action='store_true', help='is split?')
    parser.add_argument('-F', '--fit-type', metavar='', type=str, choices=('stretch', 'cut'), default='stretch', help='choose fit type when the resolutions of map and image are not the same [stretch(default), cut]')

    g_args = parser.parse_args()

def get_map_texture():
    file_name = MAP_TEXTURE_NAME
    if g_args.mono:
        file_name += '_mono_' + g_args.map_model + '_minisize.png'
    else:
        file_name += '_' + g_args.map_model + '_minisize.png'

    file_path = MAP_TEXTURE_FOLDER + '/' + file_name

    if os.path.isfile(file_path):
        return cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    else:
        script_path = BAKE_MAP_SCRIPT_FOLDER + '/' + BAKE_MAP_SCRIPT

        # Find none polyfit coeffs file, need to execute fit script.
        if g_args.mono:
            subprocess.run(['python', script_path, '-M', g_args.map_model, '--mono', '--minisize'])
        else:
            subprocess.run(['python', script_path, '-M', g_args.map_model, '--minisize'])

    if os.path.isfile(file_path):
        return cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    else:
        print('Get %s error' % (file_path))
        return np.array([])

def get_map_shape(map_texture: np.ndarray):
    global g_width
    global g_height
    global g_map_shift

    if g_args.mono:
        g_width = map_texture.shape[1]*2
    else:
        g_width = map_texture.shape[1]*2 // 3
    g_height = map_texture.shape[0]*2
    g_map_shift = (g_width - g_height) // 2

def cut_image(image: np.ndarray):
    result = np.full((g_height, g_width, 4), (0, 0, 0, 255), dtype=np.uint8)

    h = image.shape[0]
    w = image.shape[1]

    copy_x_start = 0
    copy_x_end = w
    copy_y_start = 0
    copy_y_end = h

    wrap_x_start = 0
    wrap_x_end = g_width
    wrap_y_start = 0
    wrap_y_end = g_height

    if w > g_width:
        copy_x_start = (w - g_width) // 2
        copy_x_end = copy_x_start + g_width
    else:
        wrap_x_start = (g_width - w) // 2
        wrap_x_end = wrap_x_start + w
    
    if h > g_height:
        copy_y_start = (h - g_height) // 2
        copy_y_end = copy_y_start + g_height
    else:
        wrap_y_start = (g_height - h) // 2
        wrap_y_end = wrap_y_start + h
    
    result[wrap_y_start:wrap_y_end, wrap_x_start:wrap_x_end] = image[copy_y_start:copy_y_end, copy_x_start:copy_x_end]
    return result

def get_source_image():
    images = {}

    if not os.path.exists(SOURCE_IMAGE_FOLDER):
        print('Not os.path')
        return images

    image_files = []
    for file in os.listdir(SOURCE_IMAGE_FOLDER):
        ext = os.path.splitext(file)[1]
        if ext in VALID_IMAGE_FORMAT:
            image_files.append(file)
    print(image_files)
    for file in image_files:
        file_path = os.path.join(SOURCE_IMAGE_FOLDER, file)

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print('Read %s error' % (file_path))
            continue

        h = image.shape[0]
        w = image.shape[1]
        file_name = os.path.splitext(file)[0]
        if w != g_width or h != g_height:
            match g_args.fit_type:
                case 'stretch':
                    image = cv2.resize(image, (g_width, g_height), interpolation=cv2.INTER_LANCZOS4)

                case 'cut':
                    image = cut_image(image)

                case _:
                    print('Error fit type: %s' % (g_args.fit_type))
            
            file_name = file_name + '_' + g_args.fit_type

        images[file_name] = image
        print('Get %s success' % (file))

    return images

def get_map_uv(map_texture: np.ndarray):
    x, y = np.meshgrid(np.arange(g_width), np.arange(g_height))

    coef_x = (2*x) // g_width
    coef_y = (2*y) // g_height

    flip_x = (1 - 2*coef_x)*x + coef_x*(g_width - 1)
    flip_y = (1 - 2*coef_y)*y + coef_y*(g_height - 1)

    if g_map_shift >= 0 :
        coef_x = np.minimum(1, (flip_y + g_map_shift + 1) // (flip_x + 1)) # width >= height
    else:
        coef_x = np.minimum(1, (flip_x - g_map_shift + 1) // (flip_y + 1)) # width < height
    coef_y = 1 - coef_x

    sample_x = coef_x*flip_x + coef_y*(flip_y + g_map_shift)
    sample_y = coef_y*(flip_x - g_map_shift) + coef_x*flip_y

    map_data = map_texture[sample_y, sample_x]

    map_data = np.squeeze(map_data[y, x].view(np.float32))

    center_x = float(g_width // 2) - 0.5
    center_y = float(g_height // 2) - 0.5

    map_x = (x.astype(np.float32) - center_x) * map_data + center_x
    map_y = (y.astype(np.float32) - center_y) * map_data + center_y

    return map_x, map_y

def convert_image(map_texture: np.ndarray, image: np.ndarray):
    # These calculation logics are equivalent to those in shader_effect_mask_minisize.gdshader of Godot.
    if image.shape[2] == 4:
        image_b, image_g, image_r, image_a = cv2.split(image)
    elif image.shape[2] == 3:
        image_b, image_g, image_r = cv2.split(image)
        image_a = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    else:
        print("Error image shape")
        return np.array([])

    slice_width = map_texture.shape[1] // 3

    # R channel
    map_x, map_y = get_map_uv(map_texture[:, 0:slice_width])
    image_r = cv2.remap(
        image_r, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    # G channel
    map_x, map_y = get_map_uv(map_texture[:, slice_width:slice_width*2])
    image_g = cv2.remap(
        image_g, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    # B channel
    map_x, map_y = get_map_uv(map_texture[:, slice_width*2:slice_width*3])
    image_b = cv2.remap(
        image_b, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    return cv2.merge([image_b, image_g, image_r, image_a])

def convert_image_with_mono(map_texture: np.ndarray, image: np.ndarray):
    # These calculation logics are equivalent to those in shader_effect_mask_minisize.gdshader of Godot.
    map_x, map_y = get_map_uv(map_texture)

    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

def get_split_map_uv(map_texture: np.ndarray):
    x, y = np.meshgrid(np.arange(g_width), np.arange(g_height))
    _x = 2*x

    side_flag = _x // g_width
    _x = _x - side_flag*g_width

    coef_x = (2*_x) // g_width
    coef_y = (2*y) // g_height

    flip_x = (1 - 2*coef_x)*_x + coef_x*(g_width - 1)
    flip_y = (1 - 2*coef_y)*y + coef_y*(g_height - 1)

    if g_map_shift >= 0 :
        coef_x = np.minimum(1, (flip_y + g_map_shift + 1) // (flip_x + 1)) # width >= height
    else:
        coef_x = np.minimum(1, (flip_x - g_map_shift + 1) // (flip_y + 1)) # width < height
    coef_y = 1 - coef_x

    sample_x = coef_x*flip_x + coef_y*(flip_y + g_map_shift)
    sample_y = coef_y*(flip_x - g_map_shift) + coef_x*flip_y

    map_data = map_texture[sample_y, sample_x]

    map_data = np.squeeze(map_data[y, x].view(np.float32))

    center_x = float(g_width // 4) + side_flag.astype(np.float32) * float(g_width // 2) - 0.5
    center_y = float(g_height // 2) - 0.5

    map_x = (x.astype(np.float32) - center_x) * map_data + center_x
    # Remove the over-boundary sampling in the central area.
    # This is equivalent to the logic of 
    # lessThan(_uv, vec2(0.0 + f_side_flag, 0.0) and greaterThan(_uv, vec2(0.5 + f_side_flag, 1.0) in the shader.
    side_border = float(g_width // 2)
    np.putmask(map_x, (side_flag == 0) & (map_x > (side_border - 1.0)), map_x + side_border + 1)
    np.putmask(map_x, (side_flag == 1) & (map_x < side_border), map_x - side_border - 1)

    map_y = (y.astype(np.float32) - center_y) * map_data + center_y

    return map_x, map_y

def convert_split_image(map_texture: np.ndarray, image: np.ndarray):
    # These calculation logics are equivalent to those in shader_effect_mask_minisize.gdshader of Godot.
    if image.shape[2] == 4:
        image_b, image_g, image_r, image_a = cv2.split(image)
    elif image.shape[2] == 3:
        image_b, image_g, image_r = cv2.split(image)
        image_a = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    else:
        print("Error image shape")
        return np.array([])

    slice_width = map_texture.shape[1] // 3

    # R channel
    map_x, map_y = get_split_map_uv(map_texture[:, 0:slice_width])
    image_r = cv2.remap(
        image_r, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    # G channel
    map_x, map_y = get_split_map_uv(map_texture[:, slice_width:slice_width*2])
    image_g = cv2.remap(
        image_g, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    # B channel
    map_x, map_y = get_split_map_uv(map_texture[:, slice_width*2:slice_width*3])
    image_b = cv2.remap(
        image_b, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

    return cv2.merge([image_b, image_g, image_r, image_a])

def convert_split_image_with_mono(map_texture: np.ndarray, image: np.ndarray):
    # These calculation logics are equivalent to those in shader_effect_mask_minisize_split.gdshader of Godot.
    map_x, map_y = get_split_map_uv(map_texture)

    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 255)
    )

def save_image(image: np.ndarray, name: str):
    if not os.path.exists(SAVE_IMAGE_FOLDER):
        os.makedirs(SAVE_IMAGE_FOLDER)

    # Save as png
    file_full_name = name
    if g_args.mono:
        file_full_name += '_mono_' + g_args.map_model
    else:
        file_full_name += '_' + g_args.map_model

    if g_args.split:
        file_full_name += '_split'
    save_file_path = SAVE_IMAGE_FOLDER + '/' + file_full_name + '.png'
    cv2.imwrite(save_file_path, image)

    print('Save %s success' % (save_file_path))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start convert image')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parse_argument()
    map_texture = get_map_texture()
    get_map_shape(map_texture)
    print('Map shape: %dx%d' % (g_width, g_height))

    images = get_source_image()
    if len(images) > 0:
        files = images.keys()
        for file in files:
            if g_args.split:
                if g_args.mono:
                    result = convert_split_image_with_mono(map_texture, images.get(file))
                else:
                    result = convert_split_image(map_texture, images.get(file))
            else:
                if g_args.mono:
                    result = convert_image_with_mono(map_texture, images.get(file))
                else:
                    result = convert_image(map_texture, images.get(file))
            save_image(result, file)
    else:
        print('None image file was found in folder %s' % (SOURCE_IMAGE_FOLDER))

    consume_time = time.time() - start_time
    print('End convert image')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
