import os
import time
import subprocess
import shutil

# Auxiliary Variates
CUTLINE = '--------------------------------' + '--------------------------------' + '--------------------------------'

# Constants
BAKE_SCRIPT = 'bake_map.py'
SOURCE_MAP_FOLDER = 'map_texture'
DESTINATION_MAP_FOLDER = '../godot-app/map_texture'

# Setting Variates
g_screen_width = '1920'
g_screen_height = '1080'
g_save_format = 'png'

# Process Variates
g_bake_count = 0


def run_bake(m_model: str):
    global g_bake_count

    subprocess.run([
        'python',
        BAKE_SCRIPT,
        '-F', g_save_format,
        '-R', g_screen_width, g_screen_height,
        '-M', m_model,
        '--minisize'
    ])

    g_bake_count += 2
    print('Complete baking map with %s' % (m_model))

def run_bake_all():
    run_bake('keep_center')
    run_bake('expand_edge')

def copy_map_texture():
    if not os.path.exists(SOURCE_MAP_FOLDER):
        print('Souce path %s is invalid' % (SOURCE_MAP_FOLDER))
        return

    if not os.path.exists(DESTINATION_MAP_FOLDER):
        os.makedirs(DESTINATION_MAP_FOLDER)
    
    copy_count = 0

    for item in os.listdir(SOURCE_MAP_FOLDER):
        src = os.path.join(SOURCE_MAP_FOLDER, item)
        dest = os.path.join(DESTINATION_MAP_FOLDER, item)

        shutil.copy2(src, dest)
        copy_count += 1
        print('Complete copy %s [%d/%d]' % (item, copy_count, g_bake_count))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start all')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_bake_all()
    copy_map_texture()

    consume_time = time.time() - start_time
    print('End all')
    print('Consume total time: %.4fs' % (consume_time))
    print(CUTLINE)
