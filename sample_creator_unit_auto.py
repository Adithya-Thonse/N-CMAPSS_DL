## Import libraries in python
import gc
import argparse
import os
import numpy as np
import pandas as pd
import random
from os.path import basename as opb, splitext as ops

from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen
from glob import glob

seed = 0
random.seed(0)
np.random.seed(seed)


current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
# data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='window length', required=True)
    parser.add_argument('-s', type=int, default=10, help='stride of window')
    parser.add_argument('--sampling', type=int, default=10, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('--test', type=int, default='non', help='select train or test, if it is zero, then extract samples from the engines used for training')

    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    sampling = args.sampling
    selector = args.test

    file_devtest_df = pd.read_csv("File_DevUnits_TestUnits.csv")
    for data_filepath in glob('../dataset/*'):
        # Load data
        '''
        W: operative conditions (Scenario descriptors)
        X_s: measured signals
        X_v: virtual sensors
        T(theta): engine health parameters
        Y: RUL [in cycles]
        A: auxiliary data
        '''

        df_all = df_all_creator(data_filepath, sampling)

        '''
        Split dataframe into Train and Test
        Training units: 2, 5, 10, 16, 18, 20
        Test units: 11, 14, 15        
        
        ,File,Dev Units,Test Units
        0,dataset/N-CMAPSS_DS01-005.h5,[1 2 3 4 5 6],[ 7  8  9 10]
        1,dataset/N-CMAPSS_DS04.h5,[1 2 3 4 5 6],[ 7  8  9 10]
        2,dataset/N-CMAPSS_DS08a-009.h5,[1 2 3 4 5 6 7 8 9],[10 11 12 13 14 15]
        3,dataset/N-CMAPSS_DS05.h5,[1 2 3 4 5 6],[ 7  8  9 10]
        4,dataset/N-CMAPSS_DS02-006.h5,[ 2  5 10 16 18 20],[11 14 15]
        5,dataset/N-CMAPSS_DS08c-008.h5,[1 2 3 4 5 6],[ 7  8  9 10]
        6,dataset/N-CMAPSS_DS03-012.h5,[1 2 3 4 5 6 7 8 9],[10 11 12 13 14 15]
        7,dataset/N-CMAPSS_DS07.h5,[1 2 3 4 5 6],[ 7  8  9 10]
        8,dataset/N-CMAPSS_DS06.h5,[1 2 3 4 5 6],[ 7  8  9 10]

        '''
        units_index_train = np.fromstring(
            file_devtest_df[file_devtest_df.File==opb(data_filepath)]["Dev Units"].values[0][1:-1],
            dtype=np.float, sep=' ').tolist()
        units_index_test = np.fromstring(
            file_devtest_df[file_devtest_df.File==opb(data_filepath)]["Test Units"].values[0][1:-1],
            dtype=np.float, sep=' ').tolist()

        # units = list(np.unique(df_A['unit']))
        # units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
        # units_index_test = [11.0, 14.0, 15.0]

        print("units_index_train", units_index_train)
        print("units_index_test", units_index_test)

        # if any(int(idx) == unit_index for idx in units_index_train):
        #     df_train = df_train_creator(df_all, units_index_train)
        #     print(df_train)
        #     print(df_train.columns)
        #     print("num of inputs: ", len(df_train.columns) )
        #     df_test = pd.DataFrame()
        #
        # else :
        #     df_test = df_test_creator(df_all, units_index_test)
        #     print(df_test)
        #     print(df_test.columns)
        #     print("num of inputs: ", len(df_test.columns))
        #     df_train = pd.DataFrame()

        df_train = df_train_creator(df_all, units_index_train)
        # print(df_train)
        print(df_train.columns)
        print("num of inputs: ", len(df_train.columns) )
        df_test = df_test_creator(df_all, units_index_test)
        # print(df_test)
        print(df_test.columns)
        print("num of inputs: ", len(df_test.columns))

        del df_all
        gc.collect()
        df_all = pd.DataFrame()
        sample_dir_path = os.path.join(data_filedir, 'Samples_whole', ops(opb(data_filepath))[0])
        sample_folder = os.path.isdir(sample_dir_path)
        if not sample_folder:
            os.makedirs(sample_dir_path)
            print("created folder : ", sample_dir_path)

        cols_normalize = df_train.columns.difference(['RUL', 'unit'])
        sequence_cols = df_train.columns.difference(['RUL', 'unit'])

        if selector == 0:
            for unit_index in units_index_train:
                data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                        unit_index, sampling, stride =stride)
                data_class.seq_gen()

        else:
            for unit_index in units_index_test:
                data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                        unit_index, sampling, stride =stride)
                data_class.seq_gen()


if __name__ == '__main__':
    main()
