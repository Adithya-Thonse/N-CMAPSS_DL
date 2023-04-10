## Import libraries in python
import gc
import argparse
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import shuffle
from os.path import join as opj, basename as opb, splitext as ops, exists as ope

seed = 0
random.seed(0)
np.random.seed(seed)
from glob import glob
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from math import sqrt
# import keras
import tensorflow as tf
print("Using tensorflow: ", tf.__version__)
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import Loss
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

initializer = GlorotNormal(seed=0)
# initializer = GlorotUniform(seed=0)

from utils.dnn import one_dcnn, cudnnlstm, cudnngru, deepgrucnnfc, lankygrucnnfc, MobileNetV2, transformer
from mdcl_utils import command_display, path_arg

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
# filenames = [opb(x) for x in glob(sample_dir_path + '/*')]
filenames = ['N-CMAPSS_DS02-006', 'N-CMAPSS_DS07', 'N-CMAPSS_DS06', 'N-CMAPSS_DS01-005', 'N-CMAPSS_DS05', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS08c-008']
# Excluded Datasets: 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS04',
# filenames = ['N-CMAPSS_DS02-006']
model_temp_path = os.path.join(current_dir, 'Models', 'model.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')
log_dir = os.path.join(current_dir, 'log_dir')

pic_dir = os.path.join(current_dir, 'Figures')
file_devtest_df = pd.read_csv("File_DevUnits_TestUnits.csv")


class NASAScore(Loss):
    # initialize instance attributes
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_hat):
        error = tf.math.subtract(y_hat, y_true)
        is_negative_error = error < 0.
        temp_tensor = tf.where(
            is_negative_error, tf.math.multiply(error, -1. / 13.), tf.math.multiply(error, 0.1))
        nasascore = tf.math.reduce_mean(tf.math.expm1(temp_tensor))
        # print("NASA Score: {}".format(nasascore))
        return nasascore


def nasa_score(y_true, y_hat):
    """The original NASA score as per the original whitepaper"""
    res = 0
    for true, hat in zip(y_true, y_hat):
        subs = hat - true
        if subs < 0:
            res = res + np.exp(-subs/10)-1
        else:
            res = res + np.exp(subs/13)-1
    return res/len(y_true)

def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    '''
    load array from npz files
    '''
    filename = 'Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath = os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge (sample_dir_path, unit_num, win_len, win_stride, partition):
    logger = logging.getLogger("root.load_part_array_merge")
    sample_array_lst = []
    label_array_lst = []
    logger.info("Unit: {}".format(unit_num))
    for part in range(partition):
        logger.info("Part.{}".format(part+1))
        sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
        sample_array_lst.append(sample_array)
        label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    logger.info("sample_array.shape {} label_array.shape {}".format(
        sample_array.shape, label_array.shape))
    return sample_array, label_array


def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
    filename = 'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath = os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def shuffle_array(sample_array, label_array):
    logger = logging.getLogger("root.shuffle_array")
    ind_list = list(range(len(sample_array)))
    logger.info("ind_list before: {} ... {}".format(ind_list[:10], ind_list[-10:]))
    logger.info("Shuffling in progress")
    ind_list = shuffle(ind_list)
    logger.info("ind_list after: {} ... {}".format(ind_list[:10], ind_list[-10:]))
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label


def figsave(history, win_len, win_stride, bs, lr, sub):
    logger = logging.getLogger("root.figsave")
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    # plt.show()
    pic_path = pic_dir + "/training_w%s_s%s_bs%s_sub%s_lr%s.png" %(int(win_len), int(win_stride), int(bs), int(sub), str(lr))
    logger.info("saving file:training loss figure at {}".format(pic_path))
    fig_acc.savefig(pic_path)
    return


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def scheduler(epoch, lr):
    logger = logging.getLogger("root.scheduler")
    if epoch == 30:
        logger.info("lr decay by 10")
        return lr * 0.1
    elif epoch == 70:
        logger.info("lr decay by 10")
        return lr * 0.1
    else:
        return lr


def release_list(a):
   del a[:]
   del a


def skip_samples(sample_array, label_array, skip):
    """
    If skip=0.1
    This function skips the first 10% and last 10% of the samples for a given RUL of an engine.
    This is done because the features during the change in RUL to the next number are highly noisy for the model to predict
    """
    return_label_array = np.array([], dtype=np.float32)#.reshape(0, label_array.shape[1])
    return_sample_array = np.array([], dtype=np.float32).reshape(0, sample_array.shape[1], sample_array.shape[2])

    ruls = np.unique(label_array.astype(np.int8), return_index=True)
    # ruls_ordered = ruls[0][::-1].astype(np.float32)
    ruls_ordered_indices = ruls[1][::-1]

    split_label_array = np.split(label_array, ruls_ordered_indices[1:])
    split_sample_array = np.split(sample_array, ruls_ordered_indices[1:])

    for i in range(len(split_label_array)):
        sub_array_len = len(split_label_array[i])
        skip_len = int(skip*sub_array_len)
        return_label_array = np.concatenate((return_label_array, split_label_array[i][skip_len:-skip_len]))
        return_sample_array = np.concatenate((return_sample_array, split_sample_array[i][skip_len:-skip_len]))
    return return_sample_array, return_label_array


def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length')#, required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    # parser.add_argument('-f', type=int, default=300, help='number of filter')
    # parser.add_argument('-k', type=int, default=10, help='size of kernel')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-ep', type=int, default=1000, help='max epoch')
    parser.add_argument('-pt', type=int, default=50, help='patience')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('-skip', type=int, default=0,
                        help='% of samples in each RUL of a unit to skip from beginning and end')
    parser.add_argument('--sampling', type=int, default=10,
                        help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('-only_eval_model', type=path_arg,
                        help='Evaluate a pretrained model on the test set')
    parser.add_argument('-eval_critical_ruls', action='store_true',
                        help='Evaluate the model on only the lower half of RULs')
    parser.add_argument('-band', type=int, default=5, help="+/-band to draw in the final prediction v/s truth plots")
    parser.add_argument('-model', help="+/-band to draw in the final prediction v/s truth plots", type=str,
                        choices=["one_dcnn", "cudnnlstm", "cudnngru", "deepgrucnnfc", "lankygrucnnfc", "MobileNetV2",
                                 "transformer"],
                        default="transformer")

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    # n_filters = args.f
    # kernel_size = args.k
    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub
    skip = args.skip / 100
    sampling = args.sampling
    model= args.model
    logger = command_display("{}.lis".format(ops(opb(__file__))[0]))

    if not(args.only_eval_model and ope(args.only_eval_model)):
        model_temp_path = os.path.join(current_dir, 'Models', 'model.h5')
        amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
        rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                                   name='RMSprop')
        train_units_samples_lst = []
        train_units_labels_lst = []

        for filename in filenames:
            units_index_train = np.fromstring(
                file_devtest_df[file_devtest_df.File == filename + '.h5']["Dev Units"].values[0][1:-1],
                dtype=np.float, sep=' ').tolist()

            for index in units_index_train:
                if index < 0.:
                    continue
                sample_array, label_array = load_array(opj(sample_dir_path, filename), index, win_len, win_stride, sampling)
                # sample_array, label_array = shuffle_array(sample_array, label_array)
                # logger.info("sample_array.shape {}".format(sample_array.shape))
                # logger.info("label_array.shape {}".format(label_array.shape))
                if args.skip:
                    sample_array, label_array = skip_samples(sample_array, label_array, skip)
                sample_array = sample_array[::sub]
                label_array = label_array[::sub]
                # logger.info("sub sample_array.shape {}".format(sample_array.shape))
                # logger.info("sub label_array.shape {}".format(label_array.shape))
                train_units_samples_lst.append(sample_array)
                train_units_labels_lst.append(label_array)
                logger.info("Load Train index: {:2.1f}; Subsampled by {} to {}".format(index, sub, sample_array.shape,))

        sample_array = np.concatenate(train_units_samples_lst)
        label_array = np.concatenate(train_units_labels_lst)
        # sample_array = np.reshape(sample_array, (sample_array.shape[0], 1, 50, 20))
        logger.info("Samples are aggregated")

        release_list(train_units_samples_lst)
        release_list(train_units_labels_lst)
        train_units_samples_lst =[]
        train_units_labels_lst = []
        logger.info("Memory released")

        sample_array, label_array = shuffle_array(sample_array, label_array)
        logger.info("Samples are shuffled")
        # logger.info("Sample_array.shape {}".format(sample_array.shape))
        # logger.info("Label_array.shape {}".format(label_array.shape))
        #
        # logger.info("Train sample dtype {}".format(sample_array.dtype))
        # logger.info("Train label dtype {}".format(label_array.dtype))

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        if model == "one_dcnn":
            model = one_dcnn(n_filters, kernel_size, sample_array, initializer)
        elif model == "cudnnlstm":
            model = cudnngru(sequence_length=sample_array.shape[1], nb_features=sample_array.shape[2], initializer=initializer)
        elif model == "cudnngru":
            model = cudnngru(sequence_length=sample_array.shape[1], nb_features=sample_array.shape[2], initializer=initializer)
        elif model == "deepgrucnnfc":
            model = deepgrucnnfc(sequence_length=sample_array.shape[1], nb_features=sample_array.shape[2], initializer=initializer)
        elif model == "lankygrucnnfc":
            model = lankygrucnnfc(sequence_length=sample_array.shape[1], nb_features=sample_array.shape[2], initializer=initializer)
        elif model == "MobileNetV2":
            sample_array = sample_array.reshape((sample_array.shape[0], sample_array.shape[1], sample_array.shape[2], 1))
            model = MobileNetV2()
        elif model == "transformer":
            model = transformer(sample_array.shape[1:], head_size=256, num_heads=6, ff_dim=4, num_transformer_blocks=6,
                                mlp_units=[128], mlp_dropout=0.0, dropout=0.0,)
        # model = TD_CNNBranch(n_filters, window_length=sample_array.shape[2], n_window=1, input_features=sample_array.shape[3],
        #                                strides_len=0, kernel_size=kernel_size, n_conv_layer=4, initializer=initializer)
        model.summary(print_fn=logger.info)
        # model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=[rmse, 'mae'])
        logger.info("Beginning Model Training")
        start = time.time()
        lr_scheduler = LearningRateScheduler(scheduler)
        model.compile(loss='mse', optimizer=amsgrad, metrics=['mse', ]) # NASAScore()
        history = model.fit(
            sample_array, label_array, epochs=ep, batch_size=bs, validation_split=vs, verbose=2,
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=1, mode='min'),
                         ModelCheckpoint(model_temp_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
                         TensorBoard(log_dir=log_dir, histogram_freq=1),
                         ],)
        # TqdmCallback(verbose=2)
        # model.save(tf_temp_path,save_format='tf')
        figsave(history, win_len, win_stride, bs, lr, sub)
        logger.info("Completed Model Training")

        num_train = sample_array.shape[0]
        end = time.time()
        training_time = end - start
        logger.info("Training time:  {}".format(training_time))

    else:
        logger.info("No training will be done. Only evaluating: {}".format(args.only_eval_model))
        model_temp_path = args.only_eval_model
        model = tf.keras.models.load_model(args.only_eval_model)
    logger.info("The FLOPs is:{}".format(get_flops(model)))
    # Test (inference after training)
    start = time.time()
    output_lst = []
    truth_lst = []
    # dt_num = 0
    # for filename in [filenames[dt_num]]:
    for filename in filenames:
        units_index_test = np.fromstring(
            file_devtest_df[file_devtest_df.File == filename + '.h5']["Test Units"].values[0][1:-1],
            dtype=np.float, sep=' ').tolist()

        for index in units_index_test:
            if index < 0.:
                continue
            # logger.info("test idx:  {}".format(index))
            sample_array, label_array = load_array(opj(sample_dir_path, filename), index, win_len, win_stride, sampling)
            if args.eval_critical_ruls:
                label_array_len = int(len(label_array)/2)
                sample_array, label_array = sample_array[label_array_len:], label_array[label_array_len:]
            # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
            # logger.info("sample_array.shape {}".format(sample_array.shape))
            # logger.info("label_array.shape {}".format(label_array.shape))
            if args.skip:
                sample_array, label_array = skip_samples(sample_array, label_array, skip)
            sample_array = sample_array[::sub]
            label_array = label_array[::sub]
            # logger.info("sub sample_array.shape {}".format(sample_array.shape))
            # logger.info("sub label_array.shape {}".format(label_array.shape))
            logger.info("Test data index: {:2.1f}; Subsampled by {} to {}".format(index, sub, sample_array.shape, ))
            custom_objects = {"NASAScore": NASAScore,}
            with tf.keras.utils.custom_object_scope(custom_objects):
                estimator = load_model(model_temp_path,compile=False)
            # estimator = load_model(model_temp_path)
            if model == "MobileNetV2":
                sample_array = sample_array.reshape(
                    (sample_array.shape[0], sample_array.shape[1], sample_array.shape[2], 1))
            y_pred_test = estimator.predict(sample_array)
            output_lst.append(y_pred_test)
            truth_lst.append(label_array)

        # logger.info(output_lst[0].shape)
        # logger.info(truth_lst[0].shape)

    # logger.info(np.concatenate(output_lst).shape)
    # logger.info(np.concatenate(truth_lst).shape)

    output_array = np.concatenate(output_lst)[:, 0]
    trytg_array = np.concatenate(truth_lst)
    logger.info("Predictions: {}, Ground Truths: {}".format(output_array.shape, trytg_array.shape))
    rms = round(sqrt(mean_squared_error(output_array, trytg_array)), 2)
    nasa_score_num = nasa_score(trytg_array, output_array)

    end = time.time()
    inference_time = end - start
    num_test = output_array.shape[0]
    band = args.band
    # for filename in [filenames[dt_num]]:
    for filename in filenames:
        units_index_test = np.fromstring(
            file_devtest_df[file_devtest_df.File == filename + '.h5']["Test Units"].values[0][1:-1],
            dtype=np.float, sep=' ').tolist()
        for idx in range(len(units_index_test)):
            # logger.info(output_lst[idx])
            # logger.info(truth_lst[idx])
            fig_verify = plt.figure(figsize=(24, 10))
            plt.plot(output_lst[idx], color="green")
            plt.plot(truth_lst[idx], color="red", linewidth=2.0)
            plt.fill_between(range(len(truth_lst[idx])), truth_lst[idx]+band, truth_lst[idx]-band, linestyle=':', alpha=0.5)
            plt.title('Unit%s inference with band=+/-%s' %(str(int(units_index_test[idx])), int(band)), fontsize=30)
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('RUL', fontdict={'fontsize': 24})
            plt.xlabel('Timestamps', fontdict={'fontsize': 24})
            plt.legend(['Predicted', 'Truth'], loc='upper right', fontsize=28)
            # plt.show()
            fig_verify.savefig(pic_dir + "/%s_unit%s_test_w%s_s%s_bs%s_lr%s_sub%s_rmse-%s.png" %(
                filename, str(int(units_index_test[idx])), int(win_len), int(win_stride),
                int(bs), str(lr), int(sub), str(rms)))

    logger.info("The FLOPs is:{}".format(get_flops(model)))
    logger.info("wind length_{},  win stride_{}".format(win_len, win_stride))
    if not(args.only_eval_model and ope(args.only_eval_model)):
        logger.info("# Training samples:  {}".format(num_train))
        logger.info("Training time:  {}".format(training_time))
    logger.info("# Inference samples:  {}".format(num_test))
    logger.info("Inference time:  {}".format(inference_time))
    logger.info("Result in RMSE:  {}".format(rms))
    logger.info("Result in NASA Score:  {:.2f}".format(nasa_score_num))
    return


if __name__ == '__main__':
    main()
