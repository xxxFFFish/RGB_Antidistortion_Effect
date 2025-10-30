import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Auxiliary Variates
CUTLINE = '--------------------------------'

# Constants
DATA_R_FILE_NAME = 'distortion_raw_data_r.csv'
DATA_G_FILE_NAME = 'distortion_raw_data_g.csv'
DATA_B_FILE_NAME = 'distortion_raw_data_b.csv'
PIXEL_PITCH = 0.006885 # Millimeter dimension. This value comes from 'distortion_raw_data_r/g/b.csv'
POLYFIT_DEGREE = 9
RESULT_SAVE_NAME = 'polyfit_coeffs'


def get_distortion_raw_data(file_name=''):
    data_csv = pd.read_csv(file_name)

    field_r_array = np.array(data_csv.iloc[:, 0].values)
    tan_r_array = np.tan(np.radians(field_r_array))

    real_r_array = np.array(data_csv.iloc[:, 3].values)
    real_r_array /= PIXEL_PITCH

    return tan_r_array, real_r_array

def fit_with_polyfit(in_array, out_array):
    return np.polyfit(in_array, out_array, POLYFIT_DEGREE)

def fit_process():
    coeffs = np.zeros([POLYFIT_DEGREE + 1, 3], dtype = np.float64)

    tan_r_array, real_r_array = get_distortion_raw_data(DATA_R_FILE_NAME)
    coeffs[:, 0] = fit_with_polyfit(real_r_array, tan_r_array)

    tan_r_array, real_r_array = get_distortion_raw_data(DATA_G_FILE_NAME)
    coeffs[:, 1] = fit_with_polyfit(real_r_array, tan_r_array)

    tan_r_array, real_r_array = get_distortion_raw_data(DATA_B_FILE_NAME)
    coeffs[:, 2] = fit_with_polyfit(real_r_array, tan_r_array)
    print(coeffs)
    np.save(RESULT_SAVE_NAME + '.npy', coeffs) # Save as numpy array file

    coeffs_df = pd.DataFrame(coeffs[::-1]) # Reverse order when save as csv
    coeffs_df.columns = ['Coeffs_R', 'Coeffs_G', 'Coeffs_B']
    coeffs_df.to_csv(RESULT_SAVE_NAME + '.csv', index=False)


if __name__ == '__main__':
    print(CUTLINE)
    print('Start Fit Data Process')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fit_process()

    consume_time = time.time() - start_time
    print('End Fit Data Process')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
