import os
import time
import argparse
import cv2
import numpy as np


# Auxiliary Variates
CUTLINE = '--------------------------------'

# Constants
SAVE_IMAGE_FOLDER = 'input'

# Process Variates
g_args = None


def parse_argument():
    global g_args

    parser = argparse.ArgumentParser(description='Bake Pattern Image')
    parser.add_argument('pattern', type=str, choices=('grid', 'checker', 'net'), help='pattern type')

    parser.add_argument('-R', '--resolution', metavar=('WIDTH', 'HEIGHT'), nargs=2, type=int, default=(320, 240), help='image resolution [default: 320 240]')
    parser.add_argument('-N', '--name', metavar='', type=str, default='image', help='image file name [default: image]')

    group = parser.add_argument_group('pattern repeat parameters')
    group.add_argument('--block-size', metavar='', type=int, nargs=2, default=(8, 8), help='block size of pattern [default: 8 8]')
    group.add_argument('--signet-size', metavar='', type=int, nargs=2, default=(2, 2), help='signet size of pattern [default: 2 2]')
    group.add_argument('--signet-shift', metavar='', type=int, nargs=2, default=(0, 0), help='signet shift relative to block [default: 0 0]')

    group.add_argument('--fore-color', metavar='', type=int, nargs=4, default=(255, 255, 255, 255), help='fore pattern color by RGBA [default: 255 255 255 255]')
    group.add_argument('--back-color', metavar='', type=int, nargs=4, default=(0, 0, 0, 255), help='back pattern color by RGBA [default: 0 0 0 255]')

    group = parser.add_argument_group('pattern outline parameters')
    group.add_argument('--outline-size', metavar='', type=int, default=0, help='outline size by pixel [default: 0]')
    group.add_argument('--outline-color',metavar='', type=int, nargs=4, default=(255, 255, 255, 255), help='outline color by RGBA [default: 255 255 255 255]')

    g_args = parser.parse_args()

def draw_grid(image: np.ndarray):
    draw_color = (g_args.fore_color[2], g_args.fore_color[1], g_args.fore_color[0], g_args.fore_color[3])

    block_start_y = 0
    while block_start_y < g_args.resolution[1]:
        draw_start_y = block_start_y + g_args.signet_shift[1]
        draw_end_y = draw_start_y + g_args.signet_size[1]
        if draw_end_y > g_args.resolution[1]:
            draw_end_y = g_args.resolution[1]

        block_start_x = 0
        while block_start_x < g_args.resolution[0]:
            draw_start_x = block_start_x + g_args.signet_shift[0]
            draw_end_x = draw_start_x + g_args.signet_size[0]
            if draw_end_x > g_args.resolution[0]:
                draw_end_x = g_args.resolution[0]

            image[draw_start_y:draw_end_y, draw_start_x:draw_end_x] = draw_color
            block_start_x += g_args.block_size[0]

        block_start_y += g_args.block_size[1]

def draw_checker(image: np.ndarray):
    draw_color = (g_args.fore_color[2], g_args.fore_color[1], g_args.fore_color[0], g_args.fore_color[3])

    block_start_y = 0
    draw_flag_y = True
    while block_start_y < g_args.resolution[1]:
        block_end_y = block_start_y + g_args.block_size[1]
        if block_end_y > g_args.resolution[1]:
            block_end_y = g_args.resolution[1]

        block_start_x = 0
        draw_flag_x = draw_flag_y
        while block_start_x < g_args.resolution[0]:
            block_end_x = block_start_x + g_args.block_size[0]
            if draw_flag_x:
                if block_end_x > g_args.resolution[0]:
                    block_end_x = g_args.resolution[0]

                image[block_start_y:block_end_y, block_start_x:block_end_x] = draw_color
            draw_flag_x = not draw_flag_x
            block_start_x = block_end_x

        draw_flag_y = not draw_flag_y
        block_start_y = block_end_y

def draw_net(image: np.ndarray):
    line_color = (g_args.fore_color[2], g_args.fore_color[1], g_args.fore_color[0], g_args.fore_color[3])

    block_start = 0
    while block_start < g_args.resolution[1]:
        draw_start = block_start + g_args.signet_shift[1]
        draw_end = draw_start + g_args.signet_size[1]
        if draw_end > g_args.resolution[1]:
            draw_end = g_args.resolution[1]

        image[draw_start:draw_end, :] = line_color
        block_start += g_args.block_size[1]

    block_start = 0
    while block_start < g_args.resolution[0]:
        draw_start = block_start + g_args.signet_shift[0]
        draw_end = draw_start + g_args.signet_size[0]
        if draw_end > g_args.resolution[0]:
            draw_end = g_args.resolution[0]

        image[:, draw_start:draw_end] = line_color
        block_start += g_args.block_size[0]

def draw_outline(image: np.ndarray):
    outline_color = (g_args.outline_color[2], g_args.outline_color[1], g_args.outline_color[0], g_args.outline_color[3])
    
    image[0:g_args.outline_size, :] = outline_color
    image[g_args.resolution[1] - g_args.outline_size:g_args.resolution[1], :] = outline_color

    image[:, 0:g_args.outline_size] = outline_color
    image[:, g_args.resolution[0] - g_args.outline_size:g_args.resolution[0]] = outline_color

def bake_pattern():
    back_color = (g_args.back_color[2], g_args.back_color[1], g_args.back_color[0], g_args.back_color[3])
    image = np.full((g_args.resolution[1], g_args.resolution[0], 4), back_color, dtype=np.uint8)

    match g_args.pattern:
        case 'grid':
            draw_grid(image)

        case 'checker':
            draw_checker(image)

        case 'net':
            draw_net(image)

        case _:
            print('Error pattern type: %s' % (g_args.pattern))

    if g_args.outline_size > 0:
        draw_outline(image)

    return image

def save_image(image: np.ndarray):
    if not os.path.exists(SAVE_IMAGE_FOLDER):
        os.makedirs(SAVE_IMAGE_FOLDER)

    # Save as png
    save_file_path = SAVE_IMAGE_FOLDER + '/' + g_args.name + '.png'
    cv2.imwrite(save_file_path, image)

    print('Save %s success' % (save_file_path))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start bake pattern image')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parse_argument()

    image = bake_pattern()
    save_image(image)

    consume_time = time.time() - start_time
    print('End bake pattern image')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
