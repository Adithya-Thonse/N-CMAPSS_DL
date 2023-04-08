import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
from random import shuffle
from tqdm.keras import TqdmCallback

seed = 0
random.seed(0)
np.random.seed(seed)


# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional, GRU, \
    DepthwiseConv1D, Add,GlobalAveragePooling2D, Conv2D, ReLU, DepthwiseConv2D, LayerNormalization, MultiHeadAttention,\
    GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

initializer = GlorotNormal(seed=0)
#initializer = GlorotUniform(seed=0)


def one_dcnn(n_filters, kernel_size, input_array, initializer):

    cnn = Sequential(name='one_d_cnn')
    cnn.add(Conv1D(filters=128, kernel_size=kernel_size, kernel_initializer=initializer, padding='same',
                   input_shape=(input_array.shape[1],input_array.shape[2])))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=128, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=128, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=128, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=64, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=64, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=128, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    # cnn.add(Conv1D(filters=1, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    # cnn.add(BatchNormalization())
    # cnn.add(Activation('relu'))
    cnn.add(Flatten())
    cnn.add(Dense(100, kernel_initializer=initializer))
    cnn.add(Activation('relu'))
    cnn.add(Dense(1, kernel_initializer=initializer))
    # cnn.add(Activation("linear"))
    return cnn


'''
Define the function for generating CNN braches(heads)

'''


def CNNBranch(n_filters, window_length, input_features,
              strides_len, kernel_size, n_conv_layer):
    inputs = Input(shape=(window_length, input_features), name='input_layer')
    x = inputs
    for layer in range(n_conv_layer):
        x = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    outputs = Flatten()(x)
    #     causalcnn = Model(inputs, outputs=[outputs])
    cnnbranch = Model(inputs, outputs=outputs)
    return cnnbranch


def TD_CNNBranch(n_filters, window_length, n_window, input_features,
                 strides_len, kernel_size, n_conv_layer, initializer):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer),
                            input_shape=(n_window, window_length, input_features)))
    cnn.add(TimeDistributed(BatchNormalization()))
    cnn.add(TimeDistributed(Activation('relu')))

    if n_conv_layer == 1:
        pass

    elif n_conv_layer == 2:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    elif n_conv_layer == 3:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    elif n_conv_layer == 4:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    cnn.add(TimeDistributed(Flatten()))
    print(cnn.summary())

    return cnn


def CNNB(n_filters, lr, decay, loss,
         seq_len, input_features,
         strides_len, kernel_size,
         dilation_rates):
    inputs = Input(shape=(seq_len, input_features), name='input_layer')
    x = inputs
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   dilation_rate=dilation_rate,
                   activation='linear')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # x = Dense(7, activation='relu', name='dense_layer')(x)
    outputs = Dense(3, activation='sigmoid', name='output_layer')(x)
    causalcnn = Model(inputs, outputs=[outputs])

    return causalcnn


def multi_head_cnn(sensor_input_model, n_filters, window_length, n_window,
                   input_features, strides_len, kernel_size, n_conv_layer, initializer):
    cnn_out_list = []
    cnn_branch_list = []

    for sensor_input in sensor_input_model:
        cnn_branch_temp = TD_CNNBranch(n_filters, window_length, n_window,
                                       input_features, strides_len, kernel_size, n_conv_layer, initializer)
        cnn_out_temp = cnn_branch_temp(sensor_input)

        cnn_branch_list.append(cnn_branch_temp)
        cnn_out_list.append(cnn_out_temp)

    return cnn_out_list, cnn_branch_list


def sensor_input_model(sensor_col, n_window, window_length, input_features):
    sensor_input_model = []
    for sensor in sensor_col:
        input_temp = Input(shape=(n_window, window_length, input_features), name='%s' % sensor)
        sensor_input_model.append(input_temp)

    return sensor_input_model



def cudnnlstm(sequence_length, nb_features, initializer):
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=300, kernel_initializer= initializer,
        return_sequences=True))
    model.add(LSTM(
        units=300, kernel_initializer= initializer,
        return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(
        units=300, kernel_initializer= initializer,
        return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(1, kernel_initializer=initializer))
    model.add(Activation("relu"))

    return model

"""
def cudnngru(sequence_length, nb_features, initializer):
    model = Sequential()
    model.add(GRU(
        input_shape=(sequence_length, nb_features),
        units=1024, kernel_initializer= initializer,
        return_sequences=True))
    model.add(GRU(
        units=512, kernel_initializer= initializer,
        return_sequences=True))
    model.add(GRU(
        units=256, kernel_initializer=initializer,
        return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(GRU(
        units=128, kernel_initializer= initializer,
        return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(1, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    return model
"""

def cudnngru(sequence_length, nb_features, initializer):
    model = Sequential()
    model.add(GRU(
        input_shape=(sequence_length, nb_features), units=1000, kernel_initializer= initializer, return_sequences=True))
    # model.add(GRU(units=750, kernel_initializer= initializer, return_sequences=True))
    # model.add(GRU(units=1000, kernel_initializer= initializer, return_sequences=True))
    # model.add(GRU(units=1000, kernel_initializer= initializer, return_sequences=True))
    model.add(GRU(units=500, kernel_initializer= initializer, return_sequences=True))
    # model.add(GRU(units=500, kernel_initializer=initializer, return_sequences=False))
    model.add(Conv1D(filters=100, kernel_size=15, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=25, kernel_size=15, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    '''
    # model.add(GRU(units=256, kernel_initializer=initializer,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(GRU(units=128, kernel_initializer= initializer, return_sequences=False))
    model.add(Conv1D(filters=250, kernel_size=15, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=250, kernel_size=11, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=100, kernel_size=7, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=100, kernel_size=5, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling1D(2))
    # model.add(Dropout(0.2))
    '''
    model.add(Flatten())
    # model.add(Dense(100, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    model.add(Dense(200, kernel_initializer=initializer))
    model.add(Activation("relu"))
    # model.add(Flatten())
    model.add(Dense(1, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    return model

def deepgrucnnfc(sequence_length, nb_features, initializer):
    model = Sequential()
    model.add(GRU(
        input_shape=(sequence_length, nb_features), units=500, kernel_initializer= initializer, return_sequences=True))
    model.add(GRU(units=200, kernel_initializer= initializer, return_sequences=True))
    model.add(GRU(units=200, kernel_initializer=initializer, return_sequences=True))
    model.add(GRU(units=100, kernel_initializer=initializer, return_sequences=True))
    # model.add(GRU(units=500, kernel_initializer=initializer, return_sequences=False))
    model.add(Conv1D(filters=50, kernel_size=15, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=50, kernel_size=11, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=50, kernel_size=9, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=25, kernel_size=7, kernel_initializer=initializer, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    # model.add(Dense(100, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    model.add(Dense(100, kernel_initializer=initializer))
    model.add(Activation("relu"))
    model.add(Dense(10, kernel_initializer=initializer))
    model.add(Activation("relu"))
    # model.add(Flatten())
    model.add(Dense(1, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    return model


def lankygrucnnfc(sequence_length, nb_features, initializer):
    x = Input(shape=(sequence_length, nb_features), name='input_layer')
    x = GRU(units=512, kernel_initializer= initializer, return_sequences=True)(x)
    x = GRU(units=256, kernel_initializer=initializer, return_sequences=True)(x)
    x = GRU(units=256, kernel_initializer=initializer, return_sequences=True)(x)

    # model.add(GRU(units=500, kernel_initializer=initializer, return_sequences=False))
    x = Conv1D(filters=64, kernel_size=4, kernel_initializer=initializer, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=64, kernel_size=4, kernel_initializer=initializer, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=32, kernel_size=4, kernel_initializer=initializer, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    # model.add(Dense(100, kernel_initializer=initializer))
    # model.add(Activation("relu"))
    x = Dense(100, kernel_initializer=initializer)(x)
    x = Activation("relu")(x)
    x = Dense(10, kernel_initializer=initializer)(x)
    x = Activation("relu")(x)
    x = Dense(1, kernel_initializer=initializer)(x)
    # model.add(Activation("relu"))
    return x
def mlps(vec_len, h1, h2, h3, h4):
    '''

    '''
    model = Sequential()
    model.add(Dense(h1, activation='relu', input_shape=(vec_len,)))
    model.add(Dense(h2, activation='relu'))
    model.add(Dense(h3, activation='relu'))
    model.add(Dense(h4, activation='relu'))
    model.add(Dense(1))

    return model

def Bottleneck(x,t,filters, out_channels,stride,block_id):
    y = expansion_block(x,t,filters,block_id)
    y = depthwise_block(y,stride,block_id)
    y = projection_block(y, out_channels,block_id)
    if y.shape[-1]==x.shape[-1]:
       y = Add()([x,y])
    return y

def expansion_block(x,t,filters,block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = Conv2D(total_filters,1,padding='same',use_bias=False, name =    prefix +'expand')(x)
    x = BatchNormalization(name=prefix +'expand_bn')(x)
    x = ReLU(6,name = prefix +'expand_relu')(x)
    return x
def depthwise_block(x,stride,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3,strides=(stride,stride),padding ='same', use_bias = False, name = prefix + 'depthwise_conv')(x)
    x = BatchNormalization(name=prefix +'dw_bn')(x)
    x = ReLU(6,name = prefix +'dw_relu')(x)
    return x
def projection_block(x,out_channels,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters=out_channels,kernel_size = 1,   padding='same',use_bias=False,name= prefix + 'compress')(x)
    x = BatchNormalization(name=prefix +'compress_bn')(x)
    return x


def MobileNetV2(input_shape = (50,20, 1), n_classes=1):
    input = Input (input_shape)
    x = Conv2D(32,3,strides=(2,2),padding='same', use_bias=False)(input)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6, name='conv1_relu')(x)
    # 17 Bottlenecks
    x = depthwise_block(x,stride=1,block_id=1)
    x = projection_block(x, out_channels=16,block_id=1)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)
    x = Conv2D(filters = 1280,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
    x = BatchNormalization(name='last_bn')(x)
    x = ReLU(6,name='last_relu')(x)
    x = GlobalAveragePooling2D(name='global_average_pool')(x)
    output = Dense(n_classes)(x)
    model = Model(input, output)
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)