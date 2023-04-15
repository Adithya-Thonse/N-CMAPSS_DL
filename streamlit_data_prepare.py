import pandas as pd
import numpy as np
import os
from os.path import join as opj
from tensorflow.keras.models import load_model

def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
    filename = 'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath = opj(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

models_dir = "deployment_models"
models = ["deepgrucnnfc.h5", "gru_cnn_dc.h5", "largestcudnngru.h5", "transformer.h5"]
filenames = ['N-CMAPSS_DS02-006', 'N-CMAPSS_DS07', 'N-CMAPSS_DS06', 'N-CMAPSS_DS01-005',
             'N-CMAPSS_DS05', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS08c-008', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS04']
file_devtest_df = pd.read_csv("File_DevUnits_TestUnits.csv")

def main():
    estimator_trans = load_model(opj(models_dir, "deepgrucnnfc.h5"), compile=False)
    estimator_largestcudnn = load_model(opj(models_dir, "gru_cnn_dc.h5"), compile=False)
    estimator_grucnndc = load_model(opj(models_dir, "largestcudnngru.h5"), compile=False)
    estimator_deepgru = load_model(opj(models_dir, "transformer.h5"), compile=False)

    for filename in filenames:
        units_index_test = np.fromstring(
            file_devtest_df[file_devtest_df.File == filename + '.h5']["Test Units"].values[0][1:-1],
            dtype=float, sep=' ').tolist()
        test_units_samples_lst = []
        test_units_labels_lst = []

        for index in units_index_test:
            output_list = []
            if index < 0.:
                continue
            # logger.info("test idx:  {}".format(index))
            sample_array, label_array = load_array(opj(sample_dir_path, filename), index, win_len=50, stride=1, sampling=10)
            ruls = np.unique(label_array.astype(np.int8), return_index=True)
            # ruls_ordered = ruls[0][::-1].astype(np.float32)

            ruls_ordered_indices = ruls[1][::-1]
            temp_ruls_ordered_indices = np.append(ruls_ordered_indices, label_array.shape[0])

            for i in range(len(temp_ruls_ordered_indices) - 1):
                midpoint = int((temp_ruls_ordered_indices[i] + temp_ruls_ordered_indices[i+1])/2)

                samp_array = sample_array[midpoint]
                gt = int(label_array[midpoint])
                et = int(estimator_trans.predict(samp_array.reshape((1, 50, 20)))[0][0])
                elc = int(estimator_largestcudnn.predict(samp_array.reshape((1, 50, 20)))[0][0])
                egcd = int(estimator_grucnndc.predict(samp_array.reshape((1, 50, 20)))[0][0])
                edg = int(estimator_deepgru.predict(samp_array.reshape((1, 50, 20)))[0][0])
                output_list.append([gt, et, elc, egcd, edg]) # samp_array
            columns = ['RUL', 'Transformer', 'LargestCUDNN', 'GRUCNNDC', 'DeepGRU']#, 'SampleArray']
            pd.DataFrame(output_list, columns=columns).to_csv("prediction_csv_dir/{}_Unit_{}.csv".format(filename, int(index)))



if __name__ == '__main__':
    main()
