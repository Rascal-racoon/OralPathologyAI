# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:47 2021

@author: DeepBlue
"""
import os
import sys
import ctypes
import warnings

warnings.simplefilter('ignore', FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2:No INFO, No WARNING, 0:All display
import io
import re
import math
import random
import glob
import pickle
import time
import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageColor, ImageOps
import matplotlib.pyplot as plt
from scipy import interpolate
import PySimpleGUI as sg

from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, ReLU, Reshape, DepthwiseConv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications.vgg16 import VGG16

RADAM_DISABLE = False
try:
    import tensorflow_addons as tfa
except ModuleNotFoundError:
    RADAM_DISABLE = True

IMAGESIZE_NOTINSTALLED = False
try:
    import imagesize
except ModuleNotFoundError:
    IMAGESIZE_NOTINSTALLED = True

'''
miscellanous routines
'''


def time_stamp():
    # make time-dependent unique string
    # return: str
    return str(datetime.datetime.today())[5:19].replace('-', '').replace(' ', '').replace(':', '')


def abbreviation(string):
    # input: str
    # return: str of capital letters in string
    capital_letter_list = re.findall('[A-Z]', string)
    capitals = ''
    for c in capital_letter_list:
        capitals = capitals + c
    return capitals


def check_imagesize(folder):
    # check whether all jpg files in the folder have the same size
    # input: str
    # return: bool
    impaths = glob.glob(folder + os.sep + '*.jpg')
    widths = []
    heights = []
    if len(impaths) > 10000:
        impaths = impaths[:10000]
    for impath in impaths:
        if IMAGESIZE_NOTINSTALLED:
            (width, height) = Image.open(impath).size
        else:
            width, height = imagesize.get(impath)
        widths.append(width)
        heights.append(height)
    return len(set(widths)) == 1 and len(set(heights)) == 1, heights[0], widths[0]


def check_generator(df, datapath, output_folder, mir=False, number=10, iteration=5):
    # For generator test
    # input: dataframe (pkl), patchfolder (str), outputfolder (str), number (int), iteration(int)
    # output: patch images (jpg) in outputfolder
    gen = CustomGenerator(casewise=False, xfolder=datapath, df=df, class_num=None,
                          divr=4, div=5, lot=1, mir=mir, size_adjust=False)
    generator = gen.flowBalanced(batch_size=40)
    for i in range(iteration):
        x, y = next(generator)
        for j in range(number):
            imn = x[j * 5] * 255
            imn = imn.astype(np.uint8)
            im = Image.fromarray(imn)
            im.save(output_folder + os.sep + str(i) + str(j) + str(y[j * 5]) + '.jpg')


class CustomGenerator:
    """
    Image data generator for neural network
    params:
    (casewise): bool. If True, data is splitted into train and validation by case.
    All patches from one case are seeded as either one of train or validation.
    If False, data are randomly splitted into train or validation.
    (xfolder): str. The folder containing image data, i.e. patch folder.
    (df): pandas dataframe. Dataframe that has the patch information including
    case, coordinates, label, usage, etc.
    (class_num): int. Number of class. Must be more than 1.
    (divr): int. Ratio for 'divide-random'. Used in 'Random' seeding. If divr = 4,
    for example, 3/4 data is used for training and 1/4 is used for validation.
    (div, lot): int. Ratio used in 'Casewise' seeding. 'Casewise' is not actually casewise.
    It divided the data numerically according to df index, and takes the (lot)th fraction
    for validation, and takes the rest for training. Thus patches from some cases (at most 2)
    are put into both training and validation data.
    (mir): bool. 'MIRror'. If True, patches are randomly (50:50) flipped horizontally.
    (size_adjust): bool. This option is used for patch sets that have different heights.
    If True, patch images are aligned basally and filled by black to be class PW's
    instance param self.PATCH_HEIGHT
    methods:
    (flow): Yield a batch of image patches. Patches are randomly selected, so
    classweights are necessary for imbalanced data sets.
    -params:
        (batch_size): int. number of images for a batch.
        (data): str. If data=='train', batches are generated from the training data.
        else (i.e. 'validataion'), batches are generated from the validation data.
    (flowBalanced): Yield a batch that consists of equal number of patches from each class.
    Classweight is not necessary. Batch size should be multiple of class number.
    -params:
        same as (flow)
    (func1, 2, 3, 4): These are for size adjustment and horizontal flip of patch images.
    One of 4 functions is called depending on params (mir) and (size_adjust)
    (generate_patch): Return a patch set (tensor) and a label (one-hot-formatted numpy)
    """

    def __init__(self, casewise=False, xfolder=None, df=None, class_num=None,
                 divr=4, div=5, lot=1, mir=False, size_adjust=False):
        # train/validation = divr/1
        assert divr >= 1 and divr <= 20, '1 <= divr <=20'
        assert div > 1 and div > lot, '2 <= div, and lot <= divlot'

        def strTo_int(src):  # l convert list{str] to list[int]}
            return [int(s) for s in src]

        self.xfolder = xfolder
        self.df = df
        assert 'label' in self.df.keys(), 'No key named label in the pickle file'
        self.label = self.df['label'].values.tolist()

        self.class_num = len(set(self.label))
        if class_num != None:
            self.class_num = class_num

        idx = []  # list of index of each class
        self.idxTr = [None] * self.class_num  # list of list of Train data index
        self.idxVl = [None] * self.class_num  # List of list of Validationデ data index

        if casewise:  # divide the whole data into div, and assign lot fraction to validation. The rests are train.
            _low = len(self.df) // div * lot
            _high = len(self.df) // div * (lot + 1)
            _idx = [i for i in range(len(self.df))]
            _idxTr = _idx[:_low] + _idx[_high:]  # list of train index
            _idxVl = list(set(_idx) - set(_idxTr))  # list of validation index
            dfTr = self.df.iloc[_idxTr]  # df of train
            dfVl = self.df.iloc[_idxVl]  # df of validation

            for i in range(self.class_num):
                self.idxTr[i] = strTo_int(dfTr[dfTr['label'] == i].index.tolist())
                self.idxVl[i] = strTo_int(dfVl[dfVl['label'] == i].index.tolist())
                # assert len(self.idxTr[i]) > batch_size, 'too few data number of class {}'.format(i)

        else:  # randomwise
            for i in range(self.class_num):
                idx = strTo_int(self.df[self.df['label'] == i].index.tolist())
                # assert len(idx) > batch_size, 'too few data number of class {}'.format(i)

                self.idxTr[i] = random.sample(idx, int(len(idx) * divr / (divr + 1)))
                self.idxVl[i] = list(set(idx) - set(self.idxTr[i]))

        imgpath = glob.glob(xfolder + os.sep + '*.jpg')
        assert len(imgpath) > 0, 'No .jpg in the assiged folder.'

        (self.xsize, self.ysize) = Image.open(imgpath[0]).size

        # Notcie: Train and validation are both flipped if horizontal flip checkbox is checked.
        # Although validaion is not recommended to be transformed, I think horizontal flip would make little difference. 
        if size_adjust and mir:
            self.func = self.func4
        elif size_adjust and not mir:
            self.func = self.func3
        elif not size_adjust and mir:
            self.func = self.func2
        else:
            self.func = self.func1

    def flow(self, batch_size=40, data='train'):
        # random sampling

        assert data == 'train' or data == 'validation'
        dataIdx = []
        if data == 'train':
            for i in range(self.class_num):
                dataIdx += self.idxTr[i]
        else:  # 'validation
            for i in range(self.class_num):
                dataIdx += self.idxVl[i]

        while True:
            patchset_indices = random.sample(dataIdx, batch_size)

            X_batch, y_batch = self.generate_patch(patchset_indices)
            yield X_batch, y_batch

    def flowBalanced(self, batch_size=40, data='train'):
        # generate equal numbers of each calss as a batch

        assert batch_size % self.class_num == 0, 'batch size should be multiple of class number'

        num = batch_size // self.class_num

        idx = [None] * self.class_num
        if data == 'train':
            for i in range(self.class_num):
                idx[i] = self.idxTr[i]

        else:  # data == validation
            for i in range(self.class_num):
                idx[i] = self.idxVl[i]

        while True:
            patchset_indices = []
            for i in range(self.class_num):
                patchset_indices = patchset_indices + random.sample(idx[i], num)

            X_batch, y_batch = self.generate_patch(patchset_indices)
            yield X_batch, y_batch

    def func1(self, im):
        return np.array(im)

    def func2(self, im):
        if bool(random.randint(0, 1)):
            im = ImageOps.mirror(im)
        return np.array(im)

    def func3(self, im):
        imgn = np.zeros((self.ysize, self.xsize, 3)).astype(np.uint8)
        if im.size[1] <= self.ysize:
            imgn[-im.size[1]:, :, :] = np.array(im)  # pack to basal side
        else:
            imgn = np.array(im)[-self.ysize:, :, :]
        return imgn

    def func4(self, im):
        imgn = np.zeros((self.ysize, self.xsize, 3)).astype(np.uint8)
        if bool(random.randint(0, 1)):
            im = ImageOps.mirror(im)
        if im.size[1] <= self.ysize:
            imgn[-im.size[1]:, :, :] = np.array(im)  # pack to basal side
        else:
            imgn = np.array(im)[-self.ysize:, :, :]
        return imgn

    def generate_patch(self, patchset_indices):
        current_patchList = []
        for ci in patchset_indices:
            im = Image.open(self.xfolder + os.sep + self.df.iloc[ci].name + '.jpg')
            current_patchList.append(self.func(im))
        X_batch = (np.array(current_patchList) / 255.).astype(np.float32)
        _y = np.array([self.label[i] for i in patchset_indices]).astype(np.float32)
        y_batch = to_categorical(_y, num_classes=self.class_num)

        return X_batch, y_batch


class NeuralNetwork:
    """
    Returns a neural network object.
    """

    def __init__(self):
        pass

    def MobileNet(self, input_shape=None, classes=None, **kwargs):
        # kwargs
        alpha = kwargs.get('alpha', 1.0)

        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       alpha=alpha, include_top=False, weights='imagenet',
                                                       classes=classes)
        base_model.trainable = True
        fine_tune_at = 120
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(classes)
        model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

        return model

    def MobileNet_Like(self, input_shape=None, classes=None, **kwargs):
        # kwargs
        alpha = kwargs.get('alpha', 1.0)
        depth_multiplier = kwargs.get('depth_multiplier', 1)
        dropout = kwargs.get('dropout', 1e-3)
        include_top = kwargs.get('include_top', True)
        pooling = kwargs.get('pooling', 'average')
        additional_layer = kwargs.get('additional_layer', 0)

        def pool_layer(block=1):
            # (2,1) uneven stride
            def layer_wrapper(inp):
                assert pooling == "average" or pooling == "max"
                if pooling == "average":
                    x = AveragePooling2D((2, 2), strides=(2, 1), name='block{}_pool'.format(block))(inp)
                else:
                    x = MaxPooling2D((2, 2), strides=(2, 1), name='block{}_pool'.format(block))(inp)
                return x

            return layer_wrapper

        def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
            filters = int(filters * alpha)
            x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
            x = Conv2D(filters, kernel, padding='valid', use_bias=False,
                       strides=strides, name='conv1')(x)
            x = BatchNormalization(name='conv1_bn')(x)
            return ReLU(6., name='conv1_relu')(x)

        def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                                  depth_multiplier=1, strides=(1, 1), block_id=1):
            pointwise_conv_filters = int(pointwise_conv_filters * alpha)
            if strides == (1, 1):
                x = inputs
            else:
                x = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
            x = DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid',
                                depth_multiplier=depth_multiplier, strides=strides,
                                use_bias=False, name='conv_dw_%d' % block_id)(x)
            x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
            x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
            x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False,
                       strides=(1, 1), name='conv_pw_%d' % block_id)(x)
            x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
            return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

        def _depthwise_conv_like(inputs, pointwise_conv_filters, alpha,
                                 depth_multiplier=1, strides=(1, 1), block_id=1):
            # uneven stride version (1,2)
            # Since current tf does not support uneven stride, average pooling replaces the first conv
            # x = DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid',
            #                     depth_multiplier=depth_multiplier, strides=(2, 1), 
            #                     use_bias=False, name='conv_dw_%d' % block_id)(inputs)
            # x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
            # x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
            x = pool_layer(block=block_id)(inputs)

            pointwise_conv_filters = int(pointwise_conv_filters * alpha)
            x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False,
                       strides=strides, name='conv_pw_%d' % block_id)(x)
            x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
            return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

        ####Main of MobileNet_like####

        img_input = Input(shape=input_shape)
        x = _conv_block(img_input, 32, alpha, strides=(2, 2))
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
        x = _depthwise_conv_like(x, 64, alpha, depth_multiplier, block_id=2)
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=3)
        x = _depthwise_conv_like(x, 64, alpha, depth_multiplier, block_id=4)
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=5)
        x = _depthwise_conv_like(x, 64, alpha, depth_multiplier, block_id=6)
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=7)
        x = _depthwise_conv_like(x, 64, alpha, depth_multiplier, block_id=8)

        for i in range(additional_layer):
            x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=i + 9)

        if additional_layer:
            x = _depthwise_conv_like(x, 128, alpha, depth_multiplier, block_id=9 + additional_layer)

        if include_top:
            shape = (1, 1, int(64 * alpha))
            # shape = (1, 1, int(128 * alpha))
            x = GlobalAveragePooling2D()(x)
            x = Reshape(shape, name='reshape_1')(x)
            x = Dropout(dropout, name='dropout')(x)
            x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
            x = Reshape((classes,), name='reshape_2')(x)
            y = Activation('softmax', name='act_softmax')(x)
        else:
            if pooling == 'average':
                y = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                y = GlobalMaxPooling2D()(x)
        model = Model(inputs=img_input, outputs=y, name='MobileNet')
        return model

    def DenseNet_Like(self, input_shape=None, classes=None, **kwargs):
        # kwargs
        blocks = kwargs.get('blocks', [1, 2, 4, 3])
        growth_rate = kwargs.get('growth_rate', 10)
        compression = kwargs.get('compression', 0.5)

        def dense_block(input_tensor, input_channels, nb_blocks):
            x = input_tensor
            n_channels = input_channels
            for i in range(nb_blocks):
                main = x
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = Conv2D(128, (1, 1))(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = Conv2D(growth_rate, (3, 3), padding="same")(x)
                x = Concatenate()([main, x])
                n_channels += growth_rate
            return x, n_channels

        def transition_layer(input_tensor, input_channels):
            n_channels = int(input_channels * compression)
            x = Conv2D(n_channels, (1, 1))(input_tensor)
            x = AveragePooling2D((2, 2))(x)
            return x, n_channels

        ###main of DenseNet###
        # Set blocks=[6,12,24,16] to create original DenseNet-121
        img_input = Input(shape=input_shape)

        n = 16
        x = Conv2D(n, (1, 1))(img_input)
        # DenseBlock - TransitionLayer - DenseBlock…
        for i in range(len(blocks)):
            # Transition
            if i != 0:
                x, n = transition_layer(x, n)
            # DenseBlock
            x, n = dense_block(x, n, blocks[i])

        x = GlobalAveragePooling2D()(x)

        if classes == 2:
            y = Dense(classes, activation='sigmoid')(x)
        else:
            y = Dense(classes, activation='softmax')(x)

        model = Model(inputs=img_input, outputs=y, name='DenseNet')
        return model

    def Vgg16Net(self, input_shape=None, classes=None):
        conv_base = VGG16(weights='imagenet', include_top=False)
        img_input = Input(shape=input_shape)

        x = conv_base(img_input)
        y = Flatten()(x)
        y = Dense(units=256, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)

        if classes == 2:
            y = Dense(units=classes, activation='sigmoid')(y)
        else:
            y = Dense(units=classes, activation='softmax')(y)

        model = Model(inputs=img_input, outputs=y, name='Vgg16')

        # Freeze Conv layers except block5_conv1
        conv_base.trainable = True
        set_trainable = False

        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        return model


class TAI:
    """
    TrainAI
    Open a window to input parameters and train neural networks.
    params:
    (patch_folder): str. Path to the folder that contains the patches and preferably the label (dataframe.pkl).
    (ai_folder): str. Path to the folder where model parameters (.h5) are saved.
    (output_folder): str. Path to the folder where history.df, modelsummary.txt and learning_curve.png
    are saved.
    (img_shape): tuple (height, width, 3). Patch image shape. This value is automatically replaced if the
    patch_folder contains images of the same size. If not, this value is necessary to adjust the patch
    image size.
    (window_theme): str. PySimpleGUI's theme
    (self.init): Save the {patch_folder, ai_folder, output_folder} and use in the next run
    """

    HOMEDIRECTORY = os.path.dirname(os.path.abspath(__file__))
    default_ini = {'patch_folder_tai': HOMEDIRECTORY,
                   'ai_folder_tai': HOMEDIRECTORY,
                   'output_folder_tai': HOMEDIRECTORY}

    def __init__(self, patch_folder=None, ai_folder=None, output_folder=None, img_shape=None, window_theme='DarkBlue'):
        self.init = IniHandler('ini.pkl')
        self.init.load_ini(TAI.default_ini)

        self.patch_folder = self.init.ini['patch_folder_tai'] if patch_folder == None else patch_folder
        self.ai_folder = self.init.ini[
            'ai_folder_tai'] if ai_folder == None else ai_folder  # folder that stores AI parameters .h5
        self.output_folder = self.init.ini[
            'output_folder_tai'] if output_folder == None else output_folder  # folder that stores model.summary and learning_curve
        labelpath = 'Please select'
        if len(glob.glob(self.patch_folder + os.sep + '*.pkl')) == 1:
            labelpath = glob.glob(self.patch_folder + os.sep + '*.pkl')[0]

        self.img_shape = img_shape  # A suggested img_shape from PW. Not confiremd yet

        ###Initialization###
        epochs = 20
        batch_size = 60
        generator_list = ['RandomClassweight', 'RandomBalanced', 'CasewiseClassweight', 'CasewiseBalanced']
        DIVR = 3  # In random seeding, data is divided into DIVR
        DIV = 4  # In casewise seeding, data is divided into DIV
        LOT = 0  # and LOT th fraction is taken for validation
        MIR = False  # Mirror, Horizontal flip
        models = OrderedDict([('-RMN-', 'MobileNet_Like'), ('-RDN-', 'DenseNet_Like'), ('-RDVG-', 'Vgg16Net')])

        ###### GUI part
        sg.theme(window_theme)
        txt_margin = (24, 1)
        (s0, s1) = ((4, 1), (80, 1))

        model_checkbox_layout = [sg.Text('neural network', size=txt_margin, justification='left')]
        first_flag = True  # only the first item becomes True
        for v, k in zip(models.values(), models.keys()):
            model_checkbox_layout.append(sg.Checkbox(v, key=k, default=first_flag))
            first_flag = False

        layout = [[sg.Text('Patch folder', size=txt_margin, justification='left'),
                   sg.Input(self.patch_folder, key='-INPF-', size=s1, change_submits=True),
                   sg.FolderBrowse()],
                  [sg.Text('Label (.pkl)', size=txt_margin, justification='left'),
                   sg.Input(labelpath, key='-INLB-', size=s1, change_submits=True),
                   sg.FileBrowse(file_types=(('Pickle file', '*.pkl'),))],
                  [sg.Text('Output folder', size=txt_margin, justification='left'),
                   sg.Input(self.output_folder, key='-INOP-', size=s1, change_submits=True),
                   sg.FolderBrowse(initial_folder=self.output_folder)],
                  [sg.Text('Epoch (/10)', size=txt_margin, justification='left'),
                   sg.Input(str(epochs), key='-INEP-', size=s0, change_submits=False)],
                  [sg.Text('Batch number', size=txt_margin, justification='left'),
                   sg.Input(str(batch_size), key='-INBN-', size=s0, change_submits=False)],
                  [sg.Text('Generator', size=txt_margin, justification='left'),
                   sg.Checkbox('Random-Classweight', key='-CBRC-', default=True),
                   sg.Checkbox('Random-Balanced', key='-CBRB-', default=False),
                   sg.Checkbox('Casewise-Classweight', key='-CBCC-', default=False),
                   sg.Checkbox('Casewise-Balanced', key='-CBCB-', default=False)],
                  [sg.Text('Horizontal flip', size=txt_margin, justification='left'),
                   sg.Checkbox('True', key='-CBHP-', default=False)],
                  [sg.Text('train:validation（casewise）', size=txt_margin, justification='left'),
                   sg.Input(str(DIVR), key='-INDVR-', size=s0, change_submits=False),
                   sg.Text(': 1')],
                  [sg.Text('validation（casewise）', size=txt_margin, justification='left'),
                   sg.Input(str(DIV), key='-INDV-', size=s0, change_submits=False),
                   sg.Text('- divided'),
                   sg.Input(str(LOT + 1), key='-INLT-', size=s0, change_submits=False),
                   sg.Text('th')],
                  model_checkbox_layout,
                  [sg.Text('Optimizer', size=txt_margin, justification='left'),
                   sg.Checkbox('Adam', key='-CBADM-', default=True),
                   sg.Checkbox('RAdam', key='-CBRADAM-', default=False, disabled=RADAM_DISABLE),
                   sg.Checkbox('SGD', key='-CBSGD-', default=False)],
                  [sg.Button('Start', key='-BTNSTART-'),
                   sg.Button('End')]
                  ]

        window = sg.Window('Train AI', layout)

        while True:
            event, values = window.read()
            if event in (None, sg.WIN_CLOSED):
                break

            if event == 'End':
                self.patch_folder = values['-INPF-'].replace('/', '\\')
                self.output_folder = values['-INOP-'].replace('/', '\\')

                self.init.update_ini('patch_folder_tai', self.patch_folder)
                self.init.update_ini('ai_folder_tai', self.ai_folder)
                self.init.update_ini('output_folder_tai', self.output_folder)
                self.init.save_ini()

                break

            if event == '-BTNSTART-':
                self.patch_folder = values['-INPF-'].replace('/', '\\')
                self.output_folder = values['-INOP-'].replace('/', '\\')

                self.init.update_ini('patch_folder_tai', self.patch_folder)
                self.init.update_ini('ai_folder_tai', self.ai_folder)
                self.init.update_ini('output_folder_tai', self.output_folder)
                self.init.save_ini()

                labelpath = values['-INLB-'].replace('/', '\\')
                df = pd.read_pickle(labelpath)

                # only usage=True patches are used (if available)
                if 'usage' in df.keys():
                    df = df[df['usage'] == True]

                epochs = int(values['-INEP-'])
                batch_size = int(values['-INBN-'])

                _d = {'-CBRC-': 'RandomClassweight', '-CBRB-': 'RandomBalanced', '-CBCC-': 'CasewiseClassweight',
                      '-CBCB-': 'CasewiseBalanced'}
                generator_list = []
                for _i in _d.items():
                    if values[_i[0]]:
                        generator_list.append(_i[1])

                MIR = int(values['-CBHP-'])
                DIVR = int(values['-INDVR-'])
                DIV = int(values['-INDV-'])
                LOT = int(values['-INLT-']) - 1
                assert (DIV > LOT) and (DIV < 10)

                model_list = []
                for i, j in zip(models.values(), models.keys()):
                    if values[j]:
                        model_list.append(i)

                _d = {'-CBADM-': 'Adam', '-CBRADAM-': 'RAdam', '-CBSGD-': 'SGD'}
                optimizer_list = []
                for i in _d.items():
                    if values[i[0]]:
                        optimizer_list.append(i[1])

                msg = self.train_AI(df=df, generator_list=generator_list,
                                    network_list=model_list,
                                    optimizer_list=optimizer_list,
                                    divr=DIVR, div=DIV, lot=LOT,
                                    mir=MIR, batch_size=batch_size,
                                    epochs=epochs, classnum=None)

                sg.Popup(msg, title='Confirm         ')

            elif event == '-INPF-':
                self.patch_folder = values['-INPF-'].replace('/', '\\')
                window['-INPF-'].Update(self.patch_folder)
                try:
                    labelpath = glob.glob(self.patch_folder + os.sep + '*.pkl')[0].replace('/', '\\')
                    window['-INLB-'].Update(labelpath)
                except:
                    pass

            elif event == '-INLB-':
                labelpath = values['-INLB-'].replace('/', '\\')
                window['-INLB-'].Update(labelpath)

        window.close()
        ##### End of GUI part

    def train_AI(self, df=None,
                 network_list=['MobileNet_Like'],
                 generator_list=['RandomClassweight'],
                 optimizer_list=['Adam'],
                 divr=4, div=4, lot=1, mir=False,
                 batch_size=60, epochs=10, classnum=None) -> str:

        def make_classweight(df, classnum) -> dict:
            nonlocal generator, div, lot
            label = df['label'].tolist()
            if 'Casewise' in generator:
                low = len(label) // div * lot
                high = len(label) // div * (lot + 1)
                label = label[:low] + label[high:]
            _classweight = compute_class_weight('balanced', np.unique(label), label)[:classnum]
            classweight = dict(enumerate(_classweight))
            return classweight

        def draw_learnCurve(history_df, output_folder, nametag) -> void:
            plt.figure()
            history_df[['acc', 'val_acc']].plot()

            xtks = [i * 10 for i in range((len(history_df) - 1) // 10 + 2)]
            plt.xticks(xtks, xtks)
            ytks = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
            ytklbls = ['{:.2f}'.format(i) for i in ytks]
            plt.yticks(ytks, ytklbls)
            plt.savefig(output_folder + os.sep + nametag + '.png')
            plt.close()

        #### Main of train_AI
        # data check
        imgpath = glob.glob(self.patch_folder + os.sep + '*.jpg')
        if len(imgpath) < 100:  # magic number.
            msg = 'This folder does not seem to contain patches (too few jpg files).'
            return msg

        imgcheck, height, width = check_imagesize(self.patch_folder)
        if imgcheck:
            self.img_shape = (height, width, 3)
        else:
            msg = 'Since image sizes differ, they are adjusted to the current patch size.'

        if 'label' not in df.keys():
            msg = 'You need a dataframe file with a ''label'' key.'
            return msg

        # optional
        if classnum == None:  # One can restrict training up to classmum class. Default: all.
            classnum = len(set(df['label']))

        if batch_size % classnum != 0:
            batch_size = (batch_size // classnum) * classnum
            msg = 'Batch size is reduced to {} to be a multile of class number.'.format(batch_size)

        iteration_number = len(network_list) * len(generator_list) * len(optimizer_list)
        iteration_count = 1
        start_time = time.time()

        for network in network_list:

            nn = NeuralNetwork()
            modelplan = getattr(nn, network, None)
            if modelplan == None:
                continue
            model = modelplan(self.img_shape, classnum)

            save_folder = self.output_folder + os.sep + network + time_stamp()
            os.mkdir(save_folder)

            # save model.summary()
            with open(save_folder + os.sep + 'ModelSummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\r\n'))

            for generator in generator_list:
                # each_outputfolder = self.output_folder + os.sep +network + '_' + abbreviation(generator) + '_' + time_stamp()

                if 'Casewise' in generator:
                    train_step = len(df) * (div - 1) / div // batch_size
                    val_step = int(len(df) / div // batch_size)
                else:  # Random
                    train_step = len(df) * divr / (divr + 1) // batch_size
                    val_step = int(len(df) / (divr + 1) // batch_size)
                train_step = int(train_step // 10)  # to monitor the learning progression better

                if generator == 'RandomBalanced':
                    dataGenerator = CustomGenerator(casewise=False, xfolder=self.patch_folder,
                                                    df=df, class_num=classnum, divr=divr,
                                                    mir=mir, size_adjust=not imgcheck)
                    traingen = dataGenerator.flowBalanced(batch_size=batch_size, data='train')
                    testgen = dataGenerator.flowBalanced(batch_size=batch_size, data='validation')
                elif generator == 'RandomClassweight':
                    dataGenerator = CustomGenerator(casewise=False, xfolder=self.patch_folder,
                                                    df=df, class_num=classnum, divr=divr,
                                                    mir=mir, size_adjust=not imgcheck)
                    traingen = dataGenerator.flow(batch_size=batch_size, data='train')
                    testgen = dataGenerator.flow(batch_size=batch_size, data='validation')
                elif generator == 'CasewiseBalanced':
                    dataGenerator = CustomGenerator(casewise=True, xfolder=self.patch_folder,
                                                    df=df, class_num=classnum, div=div,
                                                    lot=lot, mir=mir, size_adjust=not imgcheck)
                    traingen = dataGenerator.flowBalanced(batch_size=batch_size, data='train')
                    testgen = dataGenerator.flowBalanced(batch_size=batch_size, data='validation')
                else:  # CasewiseClassweight
                    dataGenerator = CustomGenerator(casewise=True, xfolder=self.patch_folder,
                                                    df=df, class_num=classnum, div=div,
                                                    lot=lot, mir=mir, size_adjust=not imgcheck)
                    traingen = dataGenerator.flow(batch_size=batch_size, data='train')
                    testgen = dataGenerator.flow(batch_size=batch_size, data='validation')

                if 'Balanced' in generator:
                    classweight = None
                    msg = 'Classweight is even.'
                else:  # elif Classweight
                    classweight = make_classweight(df, classnum)
                    msg = 'Classweight {} is applied.'.format(classweight)

                for optim in optimizer_list:
                    present_time = time.time()

                    # if len(network_list)*len(generator_list)*len(optimizer_list) > 1:
                    # msg = msg + '{} {} Processing.'.format(network, generator)
                    # _title = '{} {} Processing.'.format(network, generator)
                    # sg.OneLineProgressMeter(_title, cnt + 1, len(network_list)*len(generator_list)*len(optimizer_list),
                    #                             orientation='h')

                    if optim == 'RAdam':
                        optimizer = Adam(lr=1e-3)
                        # optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)
                    elif optim == 'SGD':
                        optimizer = SGD(lr=1e-2)
                    else:
                        optimizer = Adam(lr=1e-3)

                    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

                    # model save name
                    modelpath = self.ai_folder + os.sep + network + '_' + \
                                abbreviation(generator) + '_' + abbreviation(optim) + time_stamp() + '.h5'

                    # Callbacks
                    callbacks_list = [ProgressBar(iteration_number, iteration_count, start_time, present_time),
                                      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5),
                                      ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)]

                    classweight = {0: 1., 1: 1., 2: 1., 3: 1.}
                    history = model.fit(traingen, epochs=epochs, verbose=0,
                                        steps_per_epoch=train_step,
                                        callbacks=callbacks_list,
                                        validation_data=testgen,
                                        validation_steps=val_step,
                                        class_weight=classweight)
                    history_df = pd.DataFrame(history.history)

                    nametag = network + '_' + abbreviation(generator) + '_' + optim
                    history_df.to_pickle(save_folder + os.sep + 'history_' + nametag + '.pkl')
                    draw_learnCurve(history_df, save_folder, nametag)

                    iteration_count += 1
            msg = 'AI training finished.'
            return msg


class ProgressBar(Callback):
    """
    Keras custom callback
    params:
    (total_iteration): int.
    (iteration_count): int.
    (start_time): float. value of time.time() is expected (sec).
    (present_time): float.
    """

    def __init__(self, total_iteration, iteration_count, start_time, present_time):

        self.total_iteration = total_iteration
        self.iteration_count = iteration_count

        layout = [[sg.Text('', size=(60, 1), key='-TIMETEXT-')],
                  [sg.Text('', size=(60, 1), key='-BATCHTEXT-')],
                  [sg.Output(size=(60, 10), key='-OUTPUT-')]]
        self.window = sg.Window('Training in progress.', layout)
        self.window.Finalize()

        if self.iteration_count == 1:
            self.msg = 'Processing {} of {} '.format(self.iteration_count, self.total_iteration)
        else:
            passed_time = int(present_time - start_time)
            time_per_iteration = passed_time // (self.iteration_count - 1)
            remaining_time = time_per_iteration * (self.total_iteration - self.iteration_count + 1)
            self.msg = 'Finished {} of {}.  Elapsed time: {} min Expected remaining time: {} min'. \
                format(self.iteration_count - 1, self.total_iteration,
                       round(passed_time / 60), round(remaining_time / 60))

    def on_train_begin(self, logs={}):
        self.window['-TIMETEXT-'].update(self.msg)
        self.window.refresh()

    def on_batch_end(self, batch, logs={}):
        loss = logs['loss']
        acc = logs['acc']
        msg = 'Batch {:.4g}  Loss {:.4f}  Acc {:.4f}'.format(batch, loss, acc)

        self.window['-BATCHTEXT-'].update(msg)
        self.window.refresh()

    def on_test_begin(self, logs={}):
        msg = 'Testing by validation data..'
        self.window['-BATCHTEXT-'].update(msg)
        self.window.refresh()

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs['val_loss']
        val_acc = logs['val_acc']

        msg = 'Epoch {}  Val Loss {:.4f}  Val Acc {:.4f}'.format(epoch, val_loss, val_acc)
        print(msg)

    def on_train_end(self, logs={}):
        self.window.close()


class AP:
    """
    Analyze Patch
    params:
    (wsi_name): str. 'Whole Slide Image' name, such as '000000'
    (patch_set): list of Image.
    (patch_location): tuple of (x0, y0, x1, y1). (x0, y0) is left-upper. (x1, y1) is right-lower.
    (modelpath): str. Path to the model parameters (.h5).
    (cluster_num): int. Number of tissue fragements on a virtual slide image. Used for K-means clustering.
    (weights): List of float. Used to calculated the inner product of predictions.
    """
    display_modes = ('Weight', 'MaxClass', 'Each', 'CustomWeight', 'OnImage')
    color_map = 'Reds'

    def __init__(self, wsi_name='', patch_set=[], patch_location=None, modelpath='',
                 cluster_num=1):

        # display_mode
        # Weight: default. Draw multiple heatmaps using weight matrixes (changeable)
        # MaxClass: Take the class with maximum value
        # Each: Draw separate heatmaps for each class

        self.wsi_name = wsi_name
        self.patch_set = patch_set
        self.patch_loc = pd.DataFrame(patch_location,
                                      columns=['CordX0', 'CordY0', 'CordX1', 'CordY1'])
        self.modelpath = modelpath
        # self.save_folder = savefolder
        self.cluster_num = cluster_num  # number of tissues
        self.predictions = None
        self.basal_xy = None
        self.clustered_location = None

    def check_validity(self) -> tuple((bool, str)):
        # ai model exist
        if self.patch_set == []:
            return (False, 'Patch set is vacant.')
        if not os.path.isfile(self.modelpath):
            return (False, 'Such AI does not exist.')
        self.model = load_model(self.modelpath)
        # all patch sizes are checked just in case
        for i, _patch in enumerate(self.patch_set):
            if (self.model.input_shape[2], self.model.input_shape[1]) != (_patch.size):
                return (False, 'Patch size unmatch in {}th image.'.format(i + 1))
        return (True, 'Validity check passed.')

    def calculate_prediction(self):
        self.predictions = self.model.predict(AP.convert_piltonp(self.patch_set))

        self.basal_xy = self.patch_loc[['CordX0', 'CordY0']].values.tolist()  # list of lists

        if self.cluster_num > 1:
            self.clustered_location = KMeans(n_clusters=self.cluster_num).fit_predict(self.basal_xy)
        else:  # if self.cluster_num == 1, skip KMeans clustering
            self.clustered_location = np.array([0] * len(self.basal_xy))

    def draw_heatmap(self, vertical=False, display_mode=None, weights=None,
                     default_filename='Heatmap', bary=10, barx=2, margin=10):
        # color is calculated according to weights and stored in df like as color0, color1 names
        # heat map size constants
        # bary: block height
        # barx: lock width
        # margin: space around the heatmap

        # display_mode
        # Weight: default is weights=[[0, 0, 3, 3], [0, 0, 0, 3]]
        # MaxClass: argmax([p0, p1, p2, p3])
        # Each: weights=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        # CustomWeight: PW's customweight
        # default_weights = [[0, 1, 2, 3], [0, 3, 3, 3], [0, 0, 3, 3], [0, 0, 0, 3]]

        # if weights == None:
        #     weights = default_weights
        # if display_mode == 'Each':
        #     weights = [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]] #[1, 0, 0, 0] etc give identical results
        # elif display_mode == 'MaxClass':
        #     weights = [[0, 1, 2, 3]]

        key_list = []
        for i, weight in enumerate(weights):
            if display_mode == 'MaxClass':
                clr = AP.prediction_tocolor(self.predictions, weight=weight, colormap=AP.color_map)
            else:  # 'Weight' or 'Each' or 'CustomWeight'
                clr = AP.prediction_tograd(self.predictions, weight=weight, colormap=AP.color_map)

            _key = 'color' + str(i)
            self.patch_loc[_key] = clr
            key_list.append(_key)  # used below

        color_list = [self.patch_loc[k].values.tolist() for k in key_list]

        # _check_clusterresult(df, pred)
        clusters = [[] for i in range(self.cluster_num)]
        for i in range(len(self.basal_xy)):
            _color = [x[i] for x in color_list]
            clusters[self.clustered_location[i]].append([self.basal_xy[i][0], self.basal_xy[i][1], _color])

        for i in range(self.cluster_num):
            if vertical:  # tissues are placed on the slide glass, so that the epithelium is vertical
                clusters[i] = sorted(clusters[i], key=lambda x: x[1])  # sort by y axis
            else:  # default. Tissues are placed horizontally
                clusters[i].sort()
        ysize = len(clusters) * bary + (
                    len(clusters) - 1 + 2) * margin  # margins between blocks, and head & tail margins
        xsize = max([len(x) for x in clusters]) * barx + margin * 2

        heatmap_set = []
        for j, weight in enumerate(weights):
            im = Image.new('RGB', (xsize, ysize), color='black')
            draw = ImageDraw.Draw(im)

            y = margin
            for cluster in clusters:
                x = margin
                cnt = 0

                for i in cluster:
                    clr = i[2][j][:3]  # i is list of [x, y, color(RGBA)]. [:3] is to remove the alpha channel
                    draw.rectangle((x, y, x + barx, y + bary), fill=clr)
                    x += barx
                    cnt += 1

                y += bary + margin
            # im.save(savepath)
            heatmap_set.append(im)
        return heatmap_set

    @staticmethod
    def convert_piltonp(patch_set):
        # image set (list of images) is converted to tensor
        (width, height) = patch_set[0].size
        patchsetn = np.zeros((len(patch_set), height, width, 3)).astype(np.float32)
        for i, patch in enumerate(patch_set):
            patchn = np.array(patch) / 255
            _h = patchn.shape[0]
            if _h > height:
                patchsetn[i, :, :, :] = patchn[:height, :, :]
            else:
                patchsetn[i, :height, :, :] = patchn
        return patchsetn

    @staticmethod
    def prediction_tograd(predictions, weight, colormap='Reds'):
        # predictions: [[pred0, pred1, pred2, pred3], ....]

        def calculate_density(pred, weight):
            x = 0
            for p, w in zip(pred, weight):
                x += w * p
            x /= max(weight)
            return min(x, 1)

        cmap = plt.get_cmap(colormap)
        colorlist = []
        for pred in predictions:
            psum = calculate_density(pred, weight)
            _clr = cmap(psum)
            clr = tuple([int(255 * i) for i in _clr])
            colorlist.append(clr)
        return colorlist

    @staticmethod
    def prediction_tocolor(predictions, weight, colormap='Reds'):
        cmap = plt.get_cmap(colormap)
        colorlist = []
        for pred in predictions:
            p = weight[np.argmax(pred)] / max(weight)
            _clr = cmap(p)
            clr = tuple([int(255 * i) for i in _clr])
            colorlist.append(clr)
        return colorlist


class PG:
    """
    PatchGenerator
    annot_mode: False, patch generation mode. True, annotation mode. Line color changeable and no normal line drawn.
    """
    min_dotdistance = 50  # magic number: do not take too close dots
    current_lineclr = 'black'
    confirmed_lineclr = 'red'
    generation_filename = 'guide'
    annotation_filename = 'annot'

    def __init__(self, imgpath=None, patch_height=None, patch_width=None,
                 patch_folder=None, graph_size=None, initialbutton_color=None,
                 annot_mode=False):
        self.imgpath = imgpath
        self.wsi_name = os.path.basename(self.imgpath).replace('_o.jpg', '')
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_folder = patch_folder
        self.annot_mode = annot_mode
        self.annot_name = PG.generation_filename
        if self.annot_mode:
            self.annot_name = PG.annotation_filename
        self.img = Image.open(self.imgpath)
        self.imgsize = self.img.size
        self.magnification = max(self.imgsize[0] // graph_size[0],
                                 self.imgsize[1] // graph_size[1]) + 1

        self.max_magnification = self.magnification  # start with maximum reduce factor

        self.thumbnail_size = (self.imgsize[0] // self.magnification,
                               self.imgsize[1] // self.magnification)

        self.tx = self.imgsize[0] // 2  # location of thumbnail center on original image coordinates
        self.ty = self.imgsize[1] // 2
        self.make_thumbnail()
        self.clist = None
        self.all_clear(initialbutton_color)

    def switch_mode(self, mode_idx, color=None):
        # mode_idx is currently 0 or 1, which correspondes to annot_mode False and True
        self.annot_mode = bool(mode_idx)
        if self.annot_mode:
            self.annot_name = PG.annotation_filename
            self.all_clear(color)
        else:
            self.annot_name = PG.generation_filename
            self.all_clear()

    def change_patchheight(self, patch_height):
        self.patch_height = patch_height

    def change_patchwidth(self, patch_width):
        self.patch_width = patch_width

    def change_patchfolder(self, patch_folder):
        self.patch_folder = patch_folder

    def make_thumbnail(self):
        # calculate area coordinates for cropping of original image
        x0 = min(max(0, self.tx - self.thumbnail_size[0] * self.magnification // 2),
                 self.imgsize[0] - self.thumbnail_size[0] * self.magnification // 2)
        y0 = min(max(0, self.ty - self.thumbnail_size[1] * self.magnification // 2),
                 self.imgsize[1] - self.thumbnail_size[1] * self.magnification // 2)
        x1 = min(x0 + self.thumbnail_size[0] * self.magnification, self.imgsize[0] - 1)
        y1 = min(y0 + self.thumbnail_size[1] * self.magnification, self.imgsize[1] - 1)

        img = self.img.crop((x0, y0, x1, y1))
        img = img.resize((self.thumbnail_size), Image.ANTIALIAS)
        bio = io.BytesIO()
        img.save(bio, format="PNG")

        self.thumbnail = bio.getvalue()

    @staticmethod
    def spline(x, y, point=50, deg=3):
        if len(x) == 3:  # If only 3 points, ^2 spline interpolation
            deg = 2
        tck, u = interpolate.splprep([x, y], k=deg, s=0)
        u = np.linspace(0, 1, num=point, endpoint=True)
        spline = interpolate.splev(u, tck)
        return spline[0], spline[1]

    def RX(self, x):  # 'Real X': input coordinates on thumbnail window, return coordinates on original image
        return (x - self.thumbnail_size[0] // 2) * self.magnification + self.tx

    def RY(self, y):
        return (self.thumbnail_size[1] // 2 - y) * self.magnification + self.ty

    def IX(self, x):  # 'Imaginal X': input coordinates on original image,+ return coordinates on thumbnail window
        return (x - self.tx) // self.magnification + self.thumbnail_size[0] // 2

    def IY(self, y):
        return self.thumbnail_size[1] // 2 - (y - self.ty) // self.magnification

    def normal_gradient(self, xy, reverse=False, line_counter=0):  # normal; perpendicular line
        x0, y0, x1, y1 = xy[0], xy[1], xy[2], xy[3]
        v = np.array((y0 - y1, x1 - x0))
        v /= np.linalg.norm(v)

        if self.clkwise[line_counter]:  # toggle
            v = -v
        if reverse:  # reverse to fit PIL coordinate system
            v = -v
        return v

    @staticmethod
    def within_window(m, mn, mx):  # m is squeezed between mn and mx
        n = max(mn, round(m))
        n = min(n, mx)
        return int(n)

    def move(self, x, y):
        self.tx = x
        self.ty = y
        self.make_thumbnail()

    def change_magnification(self, mag):
        self.magnification = mag

        # Recalculate tx, ty.
        self.tx = min(max(self.tx, self.thumbnail_size[0] * self.magnification // 2),
                      self.imgsize[0] - self.thumbnail_size[0] * self.magnification // 2)
        self.ty = min(max(self.ty, self.thumbnail_size[1] * self.magnification // 2),
                      self.imgsize[1] - self.thumbnail_size[1] * self.magnification // 2)
        self.make_thumbnail()

    def update(self, x, y):  # x, y are coordinates on thumbnail
        if len(self.xlist) > 0:
            if abs(self.xlist[-1] - self.RX(x)) < PG.min_dotdistance and \
                    abs(self.ylist[-1] - self.RY(y)) < PG.min_dotdistance:
                return

        self.xlist.append(self.RX(x))
        self.ylist.append(self.RY(y))
        self.xy[-1] = [self.xlist, self.ylist]

    def clear(self):
        self.xlist = []
        self.ylist = []
        self.xy[-1] = [self.xlist, self.ylist]
        msg = 'Line deleted.'
        return msg

    def all_clear(self, clr=None):
        self.xlist = []
        self.ylist = []
        self.xy = [[self.xlist, self.ylist]]
        self.clkwise = [False]
        self.clist = [PG.current_lineclr]
        if self.annot_mode:
            self.clist = [clr]
        msg = 'All cleared.'
        return msg

    def delete(self):
        msg = 'Cannot delete a non-existing point.'
        if len(self.xlist) > 0:
            del self.xlist[-1]
            del self.ylist[-1]
            self.xy[-1] = [self.xlist, self.ylist]
            msg = 'Point deleted.'
        return msg

    def next_line(self, clr=None):  # clr is used only when annot_mode
        msg = 'Cannot proceed to the next line because there is less than 2 points.'
        if len(self.xlist) > 1:  # no dot or only 1 dot is not accepted to proceed to next line
            self.xlist = []
            self.ylist = []
            self.xy.append([self.xlist, self.ylist])
            self.clkwise.append(False)
            if self.annot_mode:
                self.clist.append(clr)
            else:  # patch generation mode
                self.clist[-1] = PG.confirmed_lineclr
                self.clist.append(PG.current_lineclr)
            msg = 'A new line is initiated.'
        return msg

    def change_color(self, clr):
        self.clist[-1] = clr

    def calculate_patchlocation(self, patchnum):
        # if next_line and then finish, the last list is blank
        # if next_line and point only one dot, the last list has only one dot
        # these incomplete lines must be removed.

        if len(self.xy[-1][0]) < 2:  # dot number is zero or one
            del self.xy[-1]

        # Calculate the length of lines, so as to allot patches to each line
        length = []
        for xy in self.xy:
            xyn = np.array(xy).T
            dif = np.diff(xyn, axis=0)
            _length = np.sum(np.linalg.norm(dif, axis=-1))
            length.append(_length)

        # Calculate patch number alloted to each line
        # If patch number is too larger than line length (because line is too short),
        # patch number is forced to reduce to avoid error
        if sum(length) * 2 < patchnum:  # 2; magic number
            patchnum = sum(length) // 2
            warnings.warn('Line length was too short for patch numbers. Patch number was reduced.')

        patch_numbers = [int(round(patchnum * l / sum(length))) for l in length]
        # Adjust the patch number by subtraction for the last list
        patch_numbers[-1] = patchnum - sum(patch_numbers[:-1])

        self.location = []  # list [x0, y0, x1, y1]

        line_cnt = 0  # index to show Which line is being processed now
        for xy, each_patchnum in zip(self.xy, patch_numbers):
            if each_patchnum == 0:
                line_cnt += 1
                continue

            elif each_patchnum == 1:  # rare case when the line is very short
                x = [x for x in xy[0]]
                y = [y for y in xy[1]]
                # take the first and last dots and use the center line
                x0 = (x[0] + x[-1]) / 2
                y0 = (y[0] + y[-1]) / 2
                _xy = (float(x[0]), float(y[0]), float(x[-1]), float(y[-1]))
                grad = self.normal_gradient(_xy, reverse=True, line_counter=line_cnt)
                x1 = x0 + grad[0] * self.patch_height
                y1 = y0 + grad[1] * self.patch_height
                self.location.append((x0, y0, x1, y1))

            elif each_patchnum == 2:  # rare case when the line is very short
                x = [x for x in xy[0]]
                y = [y for y in xy[1]]
                # take the first and last dots and use the center line
                # x0 = (x[0] + x[-1]) / 2
                # y0 = (y[0] + y[-1]) / 2
                _xy = (float(x[0]), float(y[0]), float(x[-1]), float(y[-1]))
                grad = self.normal_gradient(_xy, reverse=True, line_counter=line_cnt)

                for x0, y0 in zip((x[0], x[-1]), (y[0], y[-1])):
                    x1 = x0 + grad[0] * self.patch_height
                    y1 = y0 + grad[1] * self.patch_height
                    self.location.append((x0, y0, x1, y1))

            else:
                # #convert to original coordinates
                x = [x for x in xy[0]]
                y = [y for y in xy[1]]

                if len(x) == 2:  # In case of 2 dots
                    xl = [float(round(x[0] + (x[1] - x[0]) / (each_patchnum - 1) * i)) for i in range(each_patchnum)]
                    yl = [float(round(y[0] + (y[1] - y[0]) / (each_patchnum - 1) * i)) for i in range(each_patchnum)]
                else:
                    xl, yl = PG.spline(x, y, point=each_patchnum)

                cnt = 0
                for x0, y0 in zip(xl, yl):
                    if cnt != 0:
                        if cnt < len(xl) - 1:
                            _xy = (xl[cnt - 1], yl[cnt - 1], xl[cnt + 1], yl[cnt + 1])
                        else:
                            _xy = (xl[cnt - 1], yl[cnt - 1], xl[cnt], yl[cnt])
                    else:
                        _xy = (xl[cnt], yl[cnt], xl[cnt + 1], yl[cnt + 1])

                    # Reverse so as to fit the PIL coordinate system
                    grad = self.normal_gradient(_xy, reverse=True, line_counter=line_cnt)

                    x1 = x0 + grad[0] * self.patch_height
                    y1 = y0 + grad[1] * self.patch_height

                    x0 = PG.within_window(x0, 0, self.imgsize[0])
                    x1 = PG.within_window(x1, 0, self.imgsize[0])
                    y0 = PG.within_window(y0, 0, self.imgsize[1])
                    y1 = PG.within_window(y1, 0, self.imgsize[1])

                    self.location.append((x0, y0, x1, y1))
                    cnt += 1
                line_cnt += 1

    def savepatch_makelabel(self, df, patchNumber):
        marked_imagepath = self.imgpath.replace('_o', '_i')
        caselist = list(set(df['Case'].tolist()))
        if not os.path.exists(marked_imagepath):
            msg = 'No annotated file(_i) exists.'
            return df, patchNumber, msg
        elif self.wsi_name in caselist:
            existing_patchnum = len(df[df['Case'] == self.wsi_name])
            existing_patchidx = df[df['Case'] == self.wsi_name].index.tolist()

            if existing_patchnum != len(self.location):
                _msg = 'Patches from this specific case has already been included in the current patch collections.' \
                       'The same number ({}) of existing patches can be created and overwritten.' \
                       ' Change the patch number to {} first.'
                _msg = _msg.format(existing_patchnum, existing_patchnum)
                if sg.PopupOK(_msg):
                    return df, patchNumber, msg

            else:
                _msg = 'Patches from this case {} has already been included in the current patch collection.' \
                       '{} new patches are created and overwritten on the current data.' \
                       ' If you want to make more or less patches from this case,' \
                       ' remove the existing patches first. Overwrite?'
                _msg = _msg.format(self.wsi_name, existing_patchnum)
                r = sg.PopupOKCancel(_msg, title='## Warning! ##')

            if r == 'Cancel':
                msg = 'Patch generation cancelled.'
                return df, patchNumber, msg
            else:
                patch_set = self.generate_imageset()
                label_set = self.generate_labelset(marked_imagepath)

                for i, (p, l, ep) in enumerate(zip(patch_set, label_set, existing_patchidx)):
                    p.save(self.patch_folder + os.sep + ep + '.jpg')
                    _s = pd.Series([self.wsi_name, self.location[i][0], self.location[i][1],
                                    self.location[i][2], self.location[i][3], self.patch_height,
                                    True, l[1], l[2], l[3], 0, True],
                                   name=ep, index=df.columns)

                    df.loc[ep] = _s  # replace
                    # patchNumber does not chnge

                # df = PG.make_classlabel(df) #[0:norm, 1:LGD, 2:HGD, 3:SCC]
                msg = 'Finished'
                return df, patchNumber, msg

        else:
            patch_set = self.generate_imageset()
            label_set = self.generate_labelset(marked_imagepath)

            for i, (p, l) in enumerate(zip(patch_set, label_set)):
                p.save(self.patch_folder + os.sep + str(patchNumber).zfill(6) + '.jpg')
                _s = pd.Series([self.wsi_name, self.location[i][0], self.location[i][1],
                                self.location[i][2], self.location[i][3], self.patch_height,
                                True, l[1], l[2], l[3], 0, True],
                               name=str(patchNumber).zfill(6), index=df.columns)

                df = df.append(_s)
                patchNumber += 1

        # df = PG.make_classlabel(df) #[0:norm, 1:LGD, 2:HGD, 3:SCC]
        msg = 'Finished'
        return df, patchNumber, msg

    def savepatch_only_nolabel(self, patchNumber):  ## no df needed
        """
        Use only guide annotation (only epithelium label), and make patches without label file.
        """
        patch_set = self.generate_imageset()

        for i, p in enumerate(patch_set):
            p.save(self.patch_folder + os.sep + str(patchNumber).zfill(6) + '.jpg')
            patchNumber += 1
        msg = 'Finished'
        return patchNumber, msg

    # def make_classlabel(self):
    #     # User can change how each color gives the final label. The order of color is important. Latter one may override the former if both are marked.
    #     # We use green for low grade dysplasia, red for high grade dysplasia,
    #     # and blue for cancer. Thus the default is 0-normal, 1-LGD, 2-HGD, 3-SCC
    #     # The default 'black' is dummy in our dataset here.

    #     df['label'] = 0 #clear to zero        
    #     for color in classdict.keys():           
    #         df.loc[df[color] == 1, 'label'] = classdict[color]
    #     return df

    def generate_imageset(self):
        patch_set = []
        for loc in self.location:
            patch_set.append(self.crop_patch(loc, self.img))
        return patch_set

    def generate_labelset(self, path):
        img = Image.open(path)
        label_set = []
        for loc in self.location:
            label_set.append(self.check_label(loc, img))
        del img
        return label_set

    def check_label(self, location, img):
        def is_marked(src):  # check if src contains a mark
            trgt = []  # [R,G,B](boolean)
            threshold_black = 40
            src_max = np.max(np.array(src), axis=2)
            trgt.append(np.any(src_max < threshold_black))  # whether it contains black dots (guideline for epithelium)
            # this black line value is not used in the following calculation, though just keep it for some use.

            # magic numbers. threshould to determine colors.
            color = {'red': (240, 30, 30), 'green': (30, 240, 30), 'blue': (30, 30, 240)}

            threshold = 2  # minimum necessary dots regarded as marked

            for i, clr in enumerate(color):
                c1 = np.where(src[:, :, (0 + i) % 3] >= color[clr][(0 + i) % 3], 1, 0)
                c2 = np.where(src[:, :, (1 + i) % 3] <= color[clr][(1 + i) % 3], 1, 0)
                c3 = np.where(src[:, :, (2 + i) % 3] <= color[clr][(2 + i) % 3], 1, 0)
                trgt.append((np.count_nonzero(c1 * c2 * c3) >= threshold))
            return trgt  # list of booleans [Blk, R,G,B]

        patch_img = self.crop_patch(location, img)
        mark = is_marked(np.array(patch_img))
        return mark

    def crop_patch(self, location, img):
        fill_color = 'white'
        floating_error = 1  # absorb the floating error
        (x0, y0, x1, y1) = location
        length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        rad = math.atan2(y1 - y0, x1 - x0)
        angle = math.degrees(math.pi / 2 + rad)
        dx = math.floor(abs(math.sin(rad)) * self.patch_height / 2)
        dy = math.floor(abs(math.cos(rad)) * self.patch_width / 2)

        _minx = min(x0, x1) - dx
        _miny = min(y0, y1) - dy
        _maxx = max(x0, x1) + dx
        _maxy = max(y0, y1) + dy

        _minx = PG.within_window(_minx, 0, self.imgsize[0])
        _miny = PG.within_window(_miny, 0, self.imgsize[1])
        _maxx = PG.within_window(_maxx, 0, self.imgsize[0])
        _maxy = PG.within_window(_maxy, 0, self.imgsize[1])

        img_crop = img.crop((_minx, _miny, _maxx, _maxy))
        img_rotate = img_crop.rotate(angle, expand=True, fillcolor=fill_color)

        # center = img_rotate.size
        (center_x, center_y) = (img_rotate.size[0] // 2, img_rotate.size[1] // 2)

        low_x = center_x - self.patch_width // 2
        high_x = center_x + self.patch_width // 2

        if round(length, 0) >= self.patch_height - floating_error:
            low_y = center_y - self.patch_height // 2
            high_y = center_y + self.patch_height // 2
        else:
            high_y = img_rotate.size[1] - 1
            low_y = high_y - self.patch_height
        patch_img = img_rotate.crop((low_x, low_y, high_x, high_y))
        return patch_img

    def show_patchlocation(self,
                           df):  # For checking patch location. Returns thumbnail image with patch location indicated by line
        _df = df[df['Case'] == self.wsi_name]
        _c = _df[['CordX0', 'CordY0', 'CordX1', 'CordY1']].to_numpy().tolist()
        img = Image.open(io.BytesIO(self.thumbnail))
        draw = ImageDraw.Draw(img)
        for (x0, y0, x1, y1) in _c:
            draw.line((x0 // self.magnification, y0 // self.magnification,
                       x1 // self.magnification, y1 // self.magnification), fill=(0, 0, 0))
        return img

    def save_annotation(self, folder):
        msg = 'Guideline is blank.'
        if self.xy[0][0] != []:  # something exists
            savefile = (self.xy, self.xlist, self.ylist, self.clist, self.clkwise)
            filename = folder + os.sep + self.annot_name + str(self.wsi_name) + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(savefile, f)
            msg = 'Saved as {}'.format(filename)
        return msg

    def load_annotation(self, folder):
        filename = folder + os.sep + self.annot_name + str(self.wsi_name) + '.pkl'
        with open(filename, 'rb') as f:
            savefile = pickle.load(f)
        (self.xy, self.xlist, self.ylist, self.clist, self.clkwise) = savefile
        msg = 'Loaded from {}'.format(filename)
        return msg

    def draw_annotation(self, line_width=10):
        # called only when self.annot_mode = True
        img = self.img
        draw = ImageDraw.Draw(img)

        nxy = []
        for line in self.xy:
            nx = [x for x in line[0]]
            ny = [y for y in line[1]]
            nxy.append([nx, ny])

        for xy, clr in zip(nxy, self.clist):
            if len(xy[0]) >= 4:
                xl, yl = PG.spline(xy[0], xy[1])
            else:
                xl, yl = xy[0], xy[1]

            xy_tuple = tuple([(x, y) for x, y in zip(xl, yl)])
            draw.line(xy_tuple, fill=ImageColor.getrgb(clr), width=line_width)

        img.save(self.imgpath.replace('_o.jpg', '_i.jpg'))

    def dot_number(self):
        # returns number of dots in self.xy
        return sum([len(l[0]) for l in self.xy])


class IniHandler:
    """
    save, load and update initialization file
    """

    def __init__(self, filename='ini'):
        _directory = IniHandler.to_slash(os.path.dirname(os.path.abspath(__file__)))
        self.filepath = _directory + os.sep + filename
        self.ini = {}

    @staticmethod
    def to_slash(src):
        return src.replace('/', os.sep)

    def save_ini(self):
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.ini, f)

    def load_ini(self, default_ini={}):
        if os.path.isfile(self.filepath):
            with open(self.filepath, 'rb') as f:
                self.ini = pickle.load(f)
            for k in default_ini.keys():
                if not k in self.ini.keys():
                    self.ini[k] = default_ini[k]
        else:
            self.ini = default_ini

    def update_ini(self, key, value):
        self.ini[key] = value


class ListWin:
    '''
    Display the list of images(virtual slide) in shape of button window.
    params:
    (btn_name): str. List of names displayed on the buttons.
    (btn_clr): str. List of button color list [[foreground, background]].
    (btn_availability): bool. List of button disabled.
    (msg), (title): str. Messages on the window.
    
    To do: Large (still >200 or something?) list will overflow the window. 
    '''

    def __init__(self, btn_name, btn_clr=None, btn_availability=None, msg='Select one.', title='Image list'):
        grid = 40
        col = 30
        margin = 40
        data_number = len(btn_name)
        row = data_number // col + 1
        window_width = col * grid + margin * 2
        window_height = row * grid + margin
        self.btn_name = btn_name
        if btn_clr == None:
            btn_clr = [sg.DEFAULT_BUTTON_COLOR] * data_number
        if btn_availability == None:
            btn_availability = [False] * data_number

        cnt = 0
        layout = [[sg.T(msg)]]
        for i in range(row):
            _layout = []

            for j in range(col):
                _layout.append(sg.Button(btn_name[cnt], button_color=btn_clr[cnt], disabled=btn_availability[cnt]))
                cnt += 1

                if cnt >= data_number:
                    break
            else:
                layout.append(_layout)
                continue

            layout.append(_layout)
            break
        self.subwin = sg.Window(title, layout, size=(window_width, window_height),
                                location=(0, 90), modal=True, return_keyboard_events=True)

    def start(self):
        while True:
            event, values = self.subwin.read()
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
            if event == 'Escape:27':
                event = None
                break
            if event in self.btn_name:
                break
        self.subwin.close()
        return event


class PW:  # PatchWindow
    HOMEDIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # create image_folder, patch_folder, ai_folder if not exist
    # image, patch, ai

    # Initialization
    COLUMNS = ['Case', 'CordX0', 'CordY0', 'CordX1', 'CordY1', 'height',
               'black', 'red', 'green', 'blue', 'label', 'usage']
    MODE = ['Generator', 'Annotator']
    patchnum_menulist = ['10', '50', '100', '200', '300', '400', '500', '700',
                         '1000', '1500', '2000']
    patchheight_menulist = ['100 ', '200 ', '300 ', '500 ', '1000 ', '1500 ', '2000 ']  # Space is added to discriminate from above same numbers
    patchwidth_menulist = ['20', '40', '60', '80', '100']
    insetsize_menulist = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    BTNTGL_TXT = ['ToggleDirection (c)', 'ToggleColor (c)']
    BTNFIN_TXTC = ['Collect (f)', 'Readout (f)']
    BTNFIN_TXTA = ['Analyze (g)', '']
    BTNFIN_DISABLE = [False, True]
    # layout
    menu_def = [['Action', ['Renew_i.jpg', 'Make_i.jpg', 'Make_NewPatchset', 'Add_toPatchset', 'Generate_OnlyPatch_NoLabel', 'Train_AI']],
                ['Data_management', ['Change_imagefolder', 'Change_patchfolder', 'Disuse_patches', 'Restore_patches']],
                ['Data_information', ['Image_list', 'Label_information']],
                ['Patch_properties', ['Patch_number', patchnum_menulist, 'Patch_height',
                                      patchheight_menulist, 'Patch_width', patchwidth_menulist]],
                ['Settings', ['Color_mode', ['Red,Green,Blue', 'Blk,Red,Green,Blue'],
                              'Inset_size', insetsize_menulist, 'Classlabel_parameters', 'Custom_weights']]
                ]

    default_ini = {'mode': MODE[0],
                   'number': patchnum_menulist[2],
                   'height': patchheight_menulist[1],
                   'width': patchwidth_menulist[2],
                   'image_folder': HOMEDIRECTORY + os.sep + 'image',
                   'patch_folder': HOMEDIRECTORY + os.sep + 'patch',
                   'ai_folder': HOMEDIRECTORY + os.sep + 'ai',
                   'classlabel_params': {'black': ['0', 1], 'red': ['2', 3], 'green': ['1', 2], 'blue': ['3', 4]},
                   'custom_weights': [[0, 1, 2, 3]]}

    def __init__(self):
        self.init = IniHandler('ini.pkl')
        self.init.load_ini(PW.default_ini)

        self.mode_idx = PW.MODE.index(self.init.ini['mode'])
        self.PATCH_HEIGHT = int(self.init.ini['height'])
        self.PATCH_WIDTH = int(self.init.ini['width'])
        self.PATCH_NUMBER = int(self.init.ini['number'])  # number of patches

        self.IMAGEFOLDER = self.init.ini['image_folder']
        self.PATCHFOLDER = self.init.ini['patch_folder']
        self.AIFOLDER = self.init.ini['ai_folder']
        self.CLASSLABEL_PARAMS = self.init.ini['classlabel_params']
        self.CUSTOM_WEIGHTS = self.init.ini['custom_weights']

        self.IMAGEFOLDER, self.imagepaths, self.image_counter = self.read_imagefolder(self.IMAGEFOLDER)
        self.ANNOTATIONFOLDER = self.IMAGEFOLDER + os.sep + 'annotation'
        self.init.update_ini('image_folder', self.IMAGEFOLDER)

        for fld in [self.IMAGEFOLDER, self.PATCHFOLDER, self.AIFOLDER, self.ANNOTATIONFOLDER]:
            if not os.path.isdir(fld):
                os.mkdir(fld)

        # display resolution
        user32 = ctypes.WinDLL("user32")
        user32.GetSystemMetrics.restype = ctypes.c_int32
        user32.GetSystemMetrics.argtypes = (ctypes.c_int32,)
        self.SCREEN_WIDTH = user32.GetSystemMetrics(0)
        self.SCREEN_HEIGHT = user32.GetSystemMetrics(1)

        # 定数
        self.WINDOW_LOCATION = (0, 0)
        self.WINDOW_SIZE = (self.SCREEN_WIDTH * 9 // 10, self.SCREEN_HEIGHT * 8 // 10)  # magic numbers. Changeable.
        self.INSET_SIZEFACTOR = 0.8  # initial value, can be adjusted
        self.INSET_SIZE = (
        int(self.WINDOW_SIZE[0] * self.INSET_SIZEFACTOR), int(self.WINDOW_SIZE[1] * self.INSET_SIZEFACTOR))
        self.platform = None  # PG
        self.df, msg = self.load_labelfile(self.PATCHFOLDER)
        self.patchIndexNumber = len(self.df)  # patch sequential number (0,1,2... not name), to be registered next

        self.MAGNIFICATION = ['x1', 'x2', 'x3', 'x4', 'x5']
        right_click_menu = ['Enlarge', self.MAGNIFICATION]
        self.graph = sg.Graph(self.WINDOW_SIZE, (0, 0), self.WINDOW_SIZE,
                              key='-GRAPH-', background_color='black', change_submits=True,
                              drag_submits=False, right_click_menu=right_click_menu)

        # For annotation, toggle color of the button
        # 3 or 4 color mode. Initially 3 color mode.
        self.BTN_CLR = [('white', 'red'), ('white', 'lime'), ('white', 'blue')]
        self.clr_idx = 0

        initial_btnclr = self.BTN_CLR[self.clr_idx] if self.mode_idx == 1 else sg.DEFAULT_BUTTON_COLOR
        # sizes
        (bs, s0, s1, s2, s3, s4) = ((14, 1), (10, 1), (20, 1), (40, 1), (80, 1), (120, 1))

        layout = [[sg.Menu(PW.menu_def)],
                  [sg.B(self.MODE[self.mode_idx], size=s1, key='-BTNMOD-'),
                   sg.T('', size=s0, relief='sunken', text_color='black',
                        background_color='white', key='-TXTIMG-'),
                   sg.T('width {}'.format(self.PATCH_WIDTH), size=s0, relief='sunken',
                        text_color='black', background_color='white', key='-TXTPCW-'),
                   sg.T('height {}'.format(self.PATCH_HEIGHT), size=s0, relief='sunken',
                        text_color='black', background_color='white', key='-TXTPCH-'),
                   sg.T(str(self.PATCH_NUMBER) + ' patches', size=s0, relief='sunken',
                        text_color='black', background_color='white', key='-TXTPCN-'),
                   sg.T('', size=s4, relief='sunken',
                        text_color='black', background_color='white', key='-TXTMSG-')],
                  [self.graph],
                  [sg.B('DeletePoint (z)', size=bs),
                   sg.B('DeleteLine (x)', size=bs),
                   sg.B(PW.BTNTGL_TXT[self.mode_idx], size=bs, button_color=initial_btnclr, key='-BTNTGL-'),
                   sg.B('ChangeLine (v)', size=bs),
                   sg.B('DefaultView (d)', size=bs),
                   sg.B('AllClear (q)', size=bs),
                   sg.B('LoadAnnotation (a)', size=bs, disabled=False, key='-BTNLOD-'),
                   sg.B('SaveAnnotation (s)', size=bs),
                   sg.B('PreviousImage (b)', size=bs),
                   sg.B('NextImage (n)', size=bs),
                   sg.B(PW.BTNFIN_TXTC[self.mode_idx], size=bs, key='-BTNFINC-'),
                   sg.B(PW.BTNFIN_TXTA[self.mode_idx], disabled=PW.BTNFIN_DISABLE[self.mode_idx], size=bs,
                        key='-BTNFINA-'),
                   sg.B('End (e)', size=bs)]]

        self.window = sg.Window('Thin Patcher ver 1.0', layout,
                                location=self.WINDOW_LOCATION, return_keyboard_events=True)
        self.window.Finalize()
        self.window['-GRAPH-'].Widget.bind('<Shift-1>', self.move_thumbnail)
        self.window['-GRAPH-'].Widget.bind('<MouseWheel>', self.zoom)

        # self.IMAGEFOLDER, self.imagepaths, self.image_counter = self.read_imagefolder(self.IMAGEFOLDER)
        # self.ANNOTATIONFOLDER = self.IMAGEFOLDER + os.sep + 'annotation'

        # self.init.update_ini('image_folder', self.IMAGEFOLDER)
        self.renew_image(img_name=None, image_counter=0, mode_idx=self.mode_idx)
        self.update_imagename()
        self.update_message(msg)

        while True:
            event, values = self.window.Read()

            if event in (sg.WIN_CLOSED, 'Exit'):
                break

            if event == '.':
                self.key_zoom(1)

            if event == ',':
                self.key_zoom(-1)

            if event == 'End (e)' or event == 'e':
                self.readsave_ini()
                break

            if event == '-BTNMOD-':
                self.change_mode()
                self.update_message('Mode changed.')

            if event == '-GRAPH-':
                x = values['-GRAPH-'][0]
                y = values['-GRAPH-'][1]
                if x < self.platform.thumbnail_size[0] and y < self.platform.thumbnail_size[1]:
                    self.platform.update(x, y)
                    self.draw_thumbnail()
                    self.update_message()

            if event == 'DefaultView (d)' or event == 'd':
                self.default_view()

            if event == 'DeletePoint (z)' or event == 'z':
                msg = self.platform.delete()
                self.draw_thumbnail()
                self.update_message(msg)

            if event == 'DeleteLine (x)' or event == 'x':
                msg = self.platform.clear()
                self.draw_thumbnail()
                self.update_message(msg)

            if event == 'ChangeLine (v)' or event == 'v':
                msg = self.platform.next_line(self.BTN_CLR[self.clr_idx][1])
                self.draw_thumbnail()
                self.update_message(msg)

            if event == 'AllClear (q)' or event == 'q':
                msg = self.platform.all_clear(self.BTN_CLR[self.clr_idx][1])
                self.draw_thumbnail()
                self.update_message(msg)

            if event == 'NextImage (n)' or event == 'n':
                if self.image_counter < len(self.imagepaths) - 1:
                    self.image_counter += 1
                    self.renew_image(img_name=None, image_counter=self.image_counter, mode_idx=self.mode_idx)
                    self.draw_thumbnail()
                    self.update_imagename()
                    self.update_message()
                else:
                    self.update_message('This is the last image.')

            if event == 'PreviousImage (b)' or event == 'b':
                if self.image_counter > 0:
                    self.image_counter -= 1
                    self.renew_image(img_name=None, image_counter=self.image_counter, mode_idx=self.mode_idx)
                    self.draw_thumbnail()
                    self.update_imagename()
                    self.update_message()
                else:
                    self.update_message('This is the first image.')

            if event == 'SaveAnnotation (s)' or event == 's':
                if self.check_annotationfile(self.platform.wsi_name):
                    r = sg.PopupYesNo('A previous annotation exists. Overwrite?', no_titlebar=True)
                    if r == 'Yes':
                        msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)
                        self.update_message(msg)
                else:
                    msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)
                    self.update_message(msg)

            if event == '-BTNLOD-' or event == 'a':  # i.e. load annotation
                if not self.check_annotationfile(self.platform.wsi_name):
                    self.update_message('There is no saved annotation of this case.')
                else:
                    msg = self.platform.load_annotation(self.ANNOTATIONFOLDER)
                    self.draw_thumbnail()
                    self.update_message(msg)

            if (event == '-BTNFINC-' or event == 'f') and self.mode_idx == 0:  # Collect
                if self.platform.dot_number() < 2:
                    msg = 'You need at least two dots for a guide line.'
                    self.update_message(msg, color='red')
                    continue

                jpg_exist = os.path.exists(self.imagepaths[self.platform.wsi_name].replace('_o.jpg', '_i.jpg'))
                annot_exist = self.check_annotationfile(self.platform.wsi_name, PG.annotation_filename)

                if self.platform.wsi_name in list(set(self.df_indisuse().Case)):
                    msg = 'Disused patches of this case exist. You can overwrite these after you\
                        restore disused patches, and select the same number of patches. This command\
                            should be used only when you have to make different number of patches from\
                                the disused paches.'
                    r = sg.PopupOKCancel(msg, no_titlebar=True)
                    if r == 'OK':
                        self.platform.calculate_patchlocation(self.PATCH_NUMBER)
                        # send only usage=true df, and concat with usage=false df
                        new_df, self.patchIndexNumber, msg = self.platform.savepatch_makelabel(self.df_inuse(),
                                                                                               self.patchIndexNumber)
                        self.df = pd.concat([new_df, self.df_indisuse()]).sort_index()
                        self.create_classlabel()

                        msg = self.save_labelfile()
                        self.update_message(msg)
                elif not (jpg_exist or annot_exist):  # No annotation
                    msg = 'Annotation required.'
                    self.update_message(msg)
                    continue
                elif not jpg_exist:  # annot converts to _i.jpg and saved
                    msg = self.save_annotatedimage()
                    self.update_message(msg)

                if self.check_annotationfile(self.platform.wsi_name, PG.generation_filename):
                    r = sg.PopupYesNo('A previous guideline (.pkl) exists. Overwrite?', no_titlebar=True)
                    if r == 'Yes':
                        msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)
                        self.update_message(msg)
                else:
                    msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)  # this msg is unused

                self.platform.calculate_patchlocation(self.PATCH_NUMBER)
                self.df, self.patchIndexNumber, msg = self.platform.savepatch_makelabel(self.df, self.patchIndexNumber)
                self.create_classlabel()
                msg = self.save_labelfile()
                self.update_message(msg)

            if (event == '-BTNFINC-' or event == 'f') and self.mode_idx == 1:  # Readout
                if self.check_annotationfile(self.platform.wsi_name):
                    r = sg.PopupYesNo('A previous annotation (.pkl) exists. Overwrite?', no_titlebar=True)
                    if r == 'Yes':
                        msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)
                        self.update_message(msg)
                else:
                    msg = self.platform.save_annotation(self.ANNOTATIONFOLDER)  # this msg is unused
                msg = self.save_annotatedimage()
                self.update_message(msg)

            if (event == '-BTNFINA-' or event == 'g') and self.mode_idx == 0:  # Analyze
                # There must be at least two points to indicate a guide line
                if len(self.platform.xy[0][0]) < 2:
                    self.update_message('Guidelines are necessary.', color='red')
                else:
                    heatmap_set = self.run_AI(self.platform.wsi_name, self.PATCH_NUMBER)  # list of heatmaps

            if (event == '-BTNTGL-' or event == 'c') and self.mode_idx == 0:  # generator
                self.platform.clkwise[-1] = not self.platform.clkwise[-1]
                self.draw_thumbnail()
                self.update_message()

            if (event == '-BTNTGL-' or event == 'c') and self.mode_idx == 1:
                self.clr_idx = (self.clr_idx + 1) % len(self.BTN_CLR)
                self.toggle_color(self.clr_idx)
                self.update_message()

            if event == 'Change_imagefolder':
                _input = sg.popup_get_folder('Choose the folder that contains image files.',
                                             title=event, size=s3, default_path=self.init.ini['image_folder'])
                if _input != None:
                    self.IMAGEFOLDER = _input
                    self.IMAGEFOLDER, self.imagepaths, self.image_counter = self.read_imagefolder(self.IMAGEFOLDER)
                    self.ANNOTATIONFOLDER = self.IMAGEFOLDER + os.sep + 'annotation'
                    if not os.path.isdir(self.ANNOTATIONFOLDER):
                        os.mkdir(self.ANNOTATIONFOLDER)

                    self.init.update_ini('image_folder', self.IMAGEFOLDER)
                    self.renew_image(img_name=None, image_counter=0, mode_idx=self.mode_idx)
                    self.update_imagename()

            if event == 'Change_patchfolder':
                _input = sg.popup_get_folder('Choose the folder to store patches and the label.',
                                             title=event, size=s3, default_path=self.init.ini['patch_folder'])
                if _input != None:
                    self.PATCHFOLDER = _input
                    self.platform.change_patchfolder(self.PATCHFOLDER)
                    self.df, msg = self.load_labelfile(self.PATCHFOLDER)
                    self.patchIndexNumber = len(self.df)
                    self.init.update_ini('patch_folder', self.PATCHFOLDER)
                    self.update_message(msg)

            if event in PW.patchnum_menulist:  # patch number change
                if self.PATCH_NUMBER != int(event):
                    self.PATCH_NUMBER = int(event)
                    self.update_patchnumber(self.PATCH_NUMBER)
                    self.update_message('Patch number changed.')

            if event in PW.patchheight_menulist:
                if self.PATCH_HEIGHT != int(event):
                    r = sg.PopupOKCancel('Patch height will be changed.')
                    if r == 'OK':
                        self.PATCH_HEIGHT = int(event)
                        self.platform.change_patchheight(self.PATCH_HEIGHT)
                        self.window['-TXTPCH-'].update('height {}'.format(self.PATCH_HEIGHT))
                        self.draw_thumbnail()
                        self.update_message('Notice: Patch height was changed to {}'.format(self.PATCH_HEIGHT),
                                            color='red')

            if event in PW.patchwidth_menulist:
                if self.PATCH_WIDTH != int(event):
                    r = sg.PopupOKCancel('Patch width will be changed.')
                    if r == 'OK':
                        self.PATCH_WIDTH = int(event)
                        self.platform.change_patchwidth(self.PATCH_WIDTH)
                        self.window['-TXTPCW-'].update('width {}'.format(self.PATCH_WIDTH))
                        self.update_message('Notice: Patch width was changed to {}'.format(self.PATCH_WIDTH),
                                            color='red')

            if event in PW.insetsize_menulist:
                self.INSET_SIZEFACTOR = float(event)
                self.INSET_SIZE = (
                int(self.WINDOW_SIZE[0] * self.INSET_SIZEFACTOR), int(self.WINDOW_SIZE[1] * self.INSET_SIZEFACTOR))

            if event == 'Image_list':  # show image name list
                self.imagelist_window()

            if event == 'Label_information':
                self.display_labeldetails()

            if event == 'Red,Green,Blue':
                self.BTN_CLR = [('white', 'lime'), ('white', 'red'), ('white', 'blue')]
                self.clr_idx = 0
                self.toggle_color(self.clr_idx)
                self.update_message('3 color mode initiated.')

            if event == 'Blk,Red,Green,Blue':
                self.BTN_CLR = [('white', 'black'), ('white', 'lime'), ('white', 'red'), ('white', 'blue')]
                self.clr_idx = 0
                self.toggle_color(self.clr_idx)
                self.update_message('4 color mode initiated.')

            if event == 'Classlabel_parameters':
                self.set_classlabel_parameters()

            if event == 'Custom_weights':
                self.set_customweights()

            if event in self.MAGNIFICATION:
                (_x, _y) = self.graph.ClickPosition
                mag = int(event.replace('x', ''))
                self.show_inset(_x, _y, mag)

            if event == 'Disuse_patches':
                r = sg.PopupOKCancel('Patches of a selected case are disused.\
                                     This is only to change the usage flag to False,\
                                         and can be reversed afterwards.', title='Disuse_patches')
                if r == 'OK':
                    self.disuse_patch()

            if event == 'Restore_patches':
                r = sg.PopupOKCancel('Disused patches are restored. Note that\
                                     if there are multiple sets of disused patches\
                                     of the same case, all patches are restored.', title='Restore patches')
                if r == 'OK':
                    self.restore_patch()

            if event == 'Renew_i.jpg':
                r = sg.PopupOKCancel('All annotations are converted to _i.jpg.\
                                     Existing _i.jpg are replaced.', title='Renew_i.jpeg')
                if r == 'OK':
                    msg = self.convertannot_tojpg(avoid_overwrite=False)
                    self.update_message(msg)

            if event == 'Make_i.jpg':
                r = sg.PopupOKCancel('Annotations are converted to _i.jpg, \
                                     Existing _i.jpg are kept unchanged.', title='Make_i.jpg')
                if r == 'OK':
                    msg = self.convertannot_tojpg(avoid_overwrite=True)
                    self.update_message(msg)

            if event == 'Make_NewPatchset':
                msg = 'Generate a new patch set from available annotation (_i.jpg or annot.pkl).\r\n' \
                      'Width {}  Height {}  Number {}\r\n'.format(self.PATCH_WIDTH, self.PATCH_HEIGHT,
                                                                  self.PATCH_NUMBER)
                r = sg.PopupOKCancel(msg, title='Make_NewPatchset')
                if r == 'OK':
                    jpg_imgpath = glob.glob(self.PATCHFOLDER + os.sep + '*.jpg')
                    if len(jpg_imgpath) != 0:
                        msg = '## WARNING! ## There are jpg files in {}.\r\n For safty, this procedure is cancelled.\r\n' \
                              'Change the folder or remove files from this folder, or use Add_toPatchset command.'.format(
                            self.PATCHFOLDER)
                        r = sg.PopupOK(msg)
                    else:
                        msg = self.generate_Newpatch()
                        _ = self.save_labelfile()  # _ is dummy to obtain message
                        self.patchIndexNumber = len(self.df)
                        self.update_message(msg)

            if event == 'Add_toPatchset':
                msg = 'Add patches to the current patch set using available annotation (_i.jpg or annot.pkl).\r\n' \
                      'Cases already included are ignored' \
                      'Width {}  Height {}  Number {}\r\n'.format(self.PATCH_WIDTH, self.PATCH_HEIGHT,
                                                                  self.PATCH_NUMBER)
                r = sg.PopupOKCancel(msg, title='Add_toPatchset')
                if r == 'OK':
                    msg = self.generate_Addpatch()
                    _ = self.save_labelfile()
                    self.patchIndexNumber = len(self.df)
                    self.update_message(msg)

            if event == 'Generate_OnlyPatch_NoLabel':
                msg = 'Patches will be generated WITHOUT a label. You can run this command only with guide(annotation by' \
                      'Generator)_annotation without label(annotation by Annotator). This command is intended to be used ' \
                      'when you put images of a single annotation in the assigned folder.'
                r = sg.PopupOKCancel(msg, title='Generate_OnlyPatch_NoLabel')
                if r == 'OK':
                    msg = self.generate_OnlyPatch_Nolabel()
                    self.update_message(msg)

            if event == 'Train_AI':
                tai = TAI(patch_folder=None, ai_folder=self.AIFOLDER,
                          img_shape=(self.PATCH_HEIGHT, self.PATCH_WIDTH, 3))

        self.window.close()

    def set_customweights(self):
        currentweight_txt = str(self.CUSTOM_WEIGHTS[0])[1:-1]

        checklist = [False]  # dummy

        # check
        while not (len(checklist) == len(self.CUSTOM_WEIGHTS[0]) and all(checklist)):
            txt = sg.popup_get_text('Input weights. (Current weights is {}.'.format(currentweight_txt),
                                    title='Set custom weights')
            if txt == None:
                break
            # convert to list of numbers(weights)
            textlist = txt.split(',')
            checklist = [t.strip().isnumeric() for t in textlist]

        if txt != None:
            weights = [int(t.strip()) for t in textlist]
            self.CUSTOM_WEIGHTS = [weights]  # it must be a list of lists

    def create_classlabel(self):
        # User can change how each color gives the final label. The order of color is important. Latter one may override the former if both are marked.
        # We use green for low grade dysplasia, red for high grade dysplasia,
        # and blue for cancer. Thus the default is 0-normal, 1-LGD, 2-HGD, 3-SCC
        # The default 'black' is dummy in our dataset here.

        # reformat self.CLASSLABEL_PARAMS                       
        d_params = {}
        for i in range(len(self.CLASSLABEL_PARAMS)):
            for k, v in zip(self.CLASSLABEL_PARAMS.keys(), self.CLASSLABEL_PARAMS.values()):
                if v[1] == i + 1:  # order number starts from 1
                    d_params[k] = v

        if len(self.df.index) != 0:
            self.df['label'] = 0  # clear to zero
            for color in d_params.keys():
                self.df.loc[self.df[color] == 1, 'label'] = int(d_params[color][0])

    def set_classlabel_parameters(self):
        btns = []
        s0 = (10, 1)
        # params: 'line color': [value(str), order(1-4)]
        for c, l in zip(self.CLASSLABEL_PARAMS.keys(), self.CLASSLABEL_PARAMS.values()):
            btns.append([sg.T(c, size=s0),
                         sg.Radio('0', c, default='0' == l[0], key=c + '0'),
                         sg.Radio('1', c, default='1' == l[0], key=c + '1'),
                         sg.Radio('2', c, default='2' == l[0], key=c + '2'),
                         sg.Radio('3', c, default='3' == l[0], key=c + '3'),
                         sg.Radio('4', c, default='4' == l[0], key=c + '4'),
                         sg.Spin([i for i in range(1, 5)], initial_value=l[1], change_submits=True, key=c)
                         ])
        layout = [btns, [sg.B('Set', size=s0), sg.B('Cancel', size=s0)]]
        param_window = sg.Window('Label making parameter setter', layout, modal=True)

        while True:
            event, values = param_window.Read()
            if event in (sg.WIN_CLOSED, 'Cancel'):
                break
            if event in self.CLASSLABEL_PARAMS.keys():
                x = int(values[event])
                # get the index whose params[1] = x
                for c in self.CLASSLABEL_PARAMS.keys():
                    if self.CLASSLABEL_PARAMS[c][1] == x:
                        self.CLASSLABEL_PARAMS[c][1] = self.CLASSLABEL_PARAMS[event][1]
                        self.CLASSLABEL_PARAMS[event][1] = x
                        param_window[c].update(self.CLASSLABEL_PARAMS[c][1])
                        param_window[event].update(self.CLASSLABEL_PARAMS[event][1])
            if event == 'Set':
                self.CLASSLABEL_PARAMS = {}
                for k in values.keys():
                    if values[k] and type(values[k]) == bool:
                        clr = k[:-1]
                        vlu = k[-1]
                        odr = values[clr]
                        self.CLASSLABEL_PARAMS[clr] = [vlu, odr]
                break
        param_window.close()

    def readsave_ini(self):
        # load inifile (in case it has been changed by TAI or other class)
        # update all parameters on the window
        # save ini file
        self.init.load_ini(PW.default_ini)
        self.init.update_ini('mode', self.MODE[self.mode_idx])
        self.init.update_ini('number', str(self.PATCH_NUMBER))
        self.init.update_ini('height', str(self.PATCH_HEIGHT))
        self.init.update_ini('width', str(self.PATCH_WIDTH))
        self.init.update_ini('image_folder', self.IMAGEFOLDER)
        self.init.update_ini('patch_folder', self.PATCHFOLDER)
        self.init.update_ini('ai_folder', self.AIFOLDER)
        self.init.update_ini('classlabel_params', self.CLASSLABEL_PARAMS)
        self.init.update_ini('custom_weights', self.CUSTOM_WEIGHTS)
        self.init.save_ini()

    def generate_Addpatch(self):
        cnt = 0  # number of cases that generated patches
        for i, case in enumerate(self.imagepaths.keys()):  # such as '000', '001'
            sg.OneLineProgressMeter('Generating patches (adding to the previous set)', i + 1,
                                    len(self.imagepaths.keys()), orientation='h')
            _o_path = self.IMAGEFOLDER + os.sep + case + '_o.jpg'
            _i_path = self.IMAGEFOLDER + os.sep + case + '_i.jpg'

            g_flag = self.check_annotationfile(case, annot_name='guide')  # guide.pkl
            j_flag = os.path.isfile(_i_path)  # _i.jpg
            a_flag = self.check_annotationfile(case, annot_name='annot')  # annot.pkl
            p_flag = case in set(self.df.Case)  # patch. may either usage = True or False

            if g_flag and a_flag and not j_flag and not p_flag:
                # Make _i.jpg
                _platform = PG(_o_path, None, None, None, self.WINDOW_SIZE, self.BTN_CLR[self.clr_idx][1],
                               annot_mode=True)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.draw_annotation()
                # Make patch        
                _platform = PG(_o_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.calculate_patchlocation(self.PATCH_NUMBER)
                self.df, self.patchIndexNumber, msg = _platform.savepatch_makelabel(self.df, self.patchIndexNumber)
                self.create_classlabel()
                cnt += 1

            elif g_flag and j_flag and not p_flag:
                # Make patch
                _platform = PG(_o_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.calculate_patchlocation(self.PATCH_NUMBER)
                self.df, self.patchIndexNumber, msg = _platform.savepatch_makelabel(self.df, self.patchIndexNumber)
                self.create_classlabel()
                cnt += 1

        msg = '{} patches have been generated from {} cases.'.format(len(self.df), cnt)
        return msg

    def generate_OnlyPatch_Nolabel(self):
        self.patchIndexNumber = 0
        cnt = 0
        for i, case in enumerate(self.imagepaths.keys()):  # such as '000', '001'
            sg.OneLineProgressMeter('Generating patches (new set)', i + 1, len(self.imagepaths.keys()), orientation='h')
            _o_path = self.IMAGEFOLDER + os.sep + case + '_o.jpg'
            _i_path = self.IMAGEFOLDER + os.sep + case + '_i.jpg'

            g_flag = self.check_annotationfile(case, annot_name='guide')
            j_flag = os.path.isfile(_i_path)

            if not g_flag and not j_flag:
                print(f'Case {case} was skipped because missing guide-annotation.')
                continue

            if g_flag and not j_flag:
                # Make _i.jpg
                _platform = PG(_o_path, None, None, None, self.WINDOW_SIZE,
                               initialbutton_color=self.BTN_CLR[self.clr_idx][1], annot_mode=False)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.draw_annotation()

            # Make patch
            _platform = PG(_o_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
            _platform.load_annotation(self.ANNOTATIONFOLDER)
            _platform.calculate_patchlocation(self.PATCH_NUMBER)
            self.patchIndexNumber, msg = _platform.savepatch_only_nolabel(self.patchIndexNumber)
            cnt += 1

        msg = 'f{len(self.df)} patches have been generated from {cnt} cases.'
        return msg


    def generate_Newpatch(self):
        # df clear
        self.df = pd.DataFrame(index=[], columns=PW.COLUMNS)
        self.patchIndexNumber = 0
        cnt = 0  # number of cases that generated patches

        for i, case in enumerate(self.imagepaths.keys()):  # such as '000', '001'
            sg.OneLineProgressMeter('Generating patches (new set)', i + 1, len(self.imagepaths.keys()), orientation='h')
            _o_path = self.IMAGEFOLDER + os.sep + case + '_o.jpg'
            _i_path = self.IMAGEFOLDER + os.sep + case + '_i.jpg'

            g_flag = self.check_annotationfile(case, annot_name='guide')
            j_flag = os.path.isfile(_i_path)
            a_flag = self.check_annotationfile(case, annot_name='annot')

            if g_flag and a_flag and not j_flag:
                # Make _i.jpg
                _platform = PG(_o_path, None, None, None, self.WINDOW_SIZE,
                               initialbutton_color=self.BTN_CLR[self.clr_idx][1], annot_mode=True)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.draw_annotation()
                # Make patch        
                _platform = PG(_o_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.calculate_patchlocation(self.PATCH_NUMBER)
                self.df, self.patchIndexNumber, msg = _platform.savepatch_makelabel(self.df, self.patchIndexNumber)
                self.create_classlabel()
                cnt += 1

            elif g_flag and j_flag:
                # Make patch
                _platform = PG(_o_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
                _platform.load_annotation(self.ANNOTATIONFOLDER)
                _platform.calculate_patchlocation(self.PATCH_NUMBER)
                self.df, self.patchIndexNumber, msg = _platform.savepatch_makelabel(self.df, self.patchIndexNumber)
                self.create_classlabel()
                cnt += 1

        msg = '{} patches have been generated from {} cases.'.format(len(self.df), cnt)
        return msg

    def display_labeldetails(self):
        # show dataframe details including case numbers, number of each class, disused patch numbers, etc.
        total = len(self.df)
        disused = len(self.df[self.df.usage == False])
        case = len(set(self.df.Case))
        labels = [len(self.df[self.df.label == i]) for i in range(len(set(self.df.label)))]
        msg = 'Total number of patches {}\r\n' \
              'Disused patches {}\r\n' \
              'Cases {}\r\n' \
              'Each class numbers {}'.format(total, disused, case, labels)
        sg.popup(msg, title='Label information')
        # show dataframe details including case numbers, number of each class, disused patch numbers, etc.

    def df_inuse(self):  # remove deleted data
        return self.df[self.df['usage'] == True]

    def df_indisuse(self):
        return self.df[self.df['usage'] == False]

    def move_thumbnail(self, event):
        # event.x, event.y is coordinates in WINDOW_SIZE, in tkinter
        # So convert to thumbnail coordinates in pysimplegui
        _x = event.x
        _y = self.WINDOW_SIZE[1] - event.y

        # new center xy of real image
        rx = min(max(self.platform.RX(_x), self.platform.thumbnail_size[0] * self.platform.magnification // 2),
                 self.platform.imgsize[0] - self.platform.thumbnail_size[0] * self.platform.magnification // 2)
        ry = min(max(self.platform.RY(_y), self.platform.thumbnail_size[1] * self.platform.magnification // 2),
                 self.platform.imgsize[1] - self.platform.thumbnail_size[1] * self.platform.magnification // 2)

        self.platform.move(rx, ry)
        self.draw_thumbnail()

    def zoom(self, event):
        delta = event.delta // 120
        mag_l = [i + 1 for i in range(self.platform.max_magnification)]
        mag_idx = min(max(0, mag_l.index(self.platform.magnification) + delta), len(mag_l) - 1)
        self.platform.change_magnification(mag_l[mag_idx])
        self.draw_thumbnail()

    def key_zoom(self, delta):
        mag_l = [i + 1 for i in range(self.platform.max_magnification)]
        mag_idx = min(max(0, mag_l.index(self.platform.magnification) + delta), len(mag_l) - 1)
        self.platform.change_magnification(mag_l[mag_idx])
        self.draw_thumbnail()

    def draw_thumbnail(self, erase_overflow=True):
        # erase_overflow: If True, mask the normal lines out of the image

        default_dotclr = 'blue'
        # parameters for spline line format
        splinedot_max = 500
        dc = [[default_dotclr] * len(self.platform.clist), self.platform.clist]
        ds = [10, 2]  # dot size
        lw = [1, 4]  # line width

        dot_color = dc[self.mode_idx]
        dot_size = ds[self.mode_idx]
        line_color = self.platform.clist
        line_width = lw[self.mode_idx]
        normal = (self.mode_idx == 0)

        def _draw_spline(xlist, ylist, line_cnt, dc=dot_color, lc=line_color):
            spline_pointfactor = 100  # magic number
            # large number decreases number of normal lines on thumbnail (only for display).
            # No effect on patch numbers
            for x, y in zip(xlist, ylist):
                self.graph.DrawPoint((self.platform.IX(x), self.platform.IY(y)), size=dot_size, color=dc)

            if len(xlist) < 2:
                return
            elif len(xlist) == 2:  # points must be more than 3 for spline
                xl, yl = xlist, ylist
            else:
                # calculate appropriate number of points
                xyn = np.array([xlist, ylist]).T
                dif = np.diff(xyn, axis=0)
                length = np.sum(np.linalg.norm(dif, axis=-1))
                points = int(min(max(5, length // spline_pointfactor), splinedot_max))  #

                xl, yl = PG.spline(xlist, ylist, point=points)
            before_value = (0, 0)  # coding checker gives an error without this line
            cnt = 0
            patch_h = self.PATCH_HEIGHT // self.platform.magnification

            for x, y in zip(xl, yl):

                if cnt != 0:
                    self.graph.DrawLine(before_value, (self.platform.IX(x), self.platform.IY(y)), color=lc,
                                        width=line_width)

                    if cnt < len(xl) - 1:
                        _xy = (self.platform.IX(xl[cnt - 1]), self.platform.IY(yl[cnt - 1]),
                               self.platform.IX(xl[cnt + 1]), self.platform.IY(yl[cnt + 1]))
                    else:
                        _xy = (self.platform.IX(xl[cnt - 1]), self.platform.IY(yl[cnt - 1]),
                               self.platform.IX(xl[cnt]), self.platform.IY(yl[cnt]))
                else:
                    _xy = (self.platform.IX(xl[cnt]), self.platform.IY(yl[cnt]),
                           self.platform.IX(xl[cnt + 1]), self.platform.IY(yl[cnt + 1]))

                if normal and len(xlist) >= 3:
                    grad = self.platform.normal_gradient(_xy, line_counter=line_cnt)
                    xn = self.platform.IX(x) + grad[0] * patch_h
                    yn = self.platform.IY(y) + grad[1] * patch_h
                    self.graph.DrawLine((self.platform.IX(x), self.platform.IY(y)),
                                        (xn, yn), color='cyan')

                    # erase stickedout normal lines
                    if erase_overflow:
                        self.graph.DrawRectangle((0, self.WINDOW_SIZE[1]),
                                                 (self.WINDOW_SIZE[0], self.platform.thumbnail_size[1]),
                                                 fill_color='black', line_color='black')
                        self.graph.DrawRectangle((self.platform.thumbnail_size[0], self.WINDOW_SIZE[1]),
                                                 (self.WINDOW_SIZE[0], 0),
                                                 fill_color='black', line_color='black')

                before_value = (self.platform.IX(x), self.platform.IY(y))
                cnt += 1

        # Main of def draw_thumbnail
        self.graph.Erase()
        self.graph.DrawImage(data=self.platform.thumbnail, location=(0, self.platform.thumbnail_size[1]))

        for line_cnt, (xy, dclr, lclr) in enumerate(zip(self.platform.xy, dot_color, line_color)):
            _draw_spline(xy[0], xy[1], line_cnt, dc=dclr, lc=lclr)

    def update_imagename(self):
        self.window['-TXTIMG-'].update(self.platform.wsi_name)

    def update_loadbutton(self):  # able or disalbe load_annotation button
        _disabled = not self.check_annotationfile(self.platform.wsi_name)
        self.window['-BTNLOD-'].update(disabled=_disabled)

    def update_patchnumber(self, patchnum):
        self.window['-TXTPCN-'].update(str(patchnum) + ' patches')

    def update_message(self, msg='', color='black'):
        if isinstance(msg, list):  # msg include color [message, color]
            color = msg[1]
            msg = msg[0]
        self.window['-TXTMSG-'].update(msg, text_color=color)

    def renew_image(self, img_name=None, image_counter=None, mode_idx=0):
        if img_name == None and image_counter == None:
            return self.platform.thumbnail_size
        elif img_name != None:
            _im_path = self.imagepaths[img_name]
        else:  # image_counter != None
            _im_path = list(self.imagepaths.values())[image_counter]

        sg.PopupQuickMessage('Loading..')
        if self.mode_idx == 0:
            self.platform = PG(_im_path, self.PATCH_HEIGHT, self.PATCH_WIDTH, self.PATCHFOLDER, self.WINDOW_SIZE)
        else:  # mode_idx == 1:
            self.platform = PG(_im_path, None, None, None, self.WINDOW_SIZE,
                               initialbutton_color=self.BTN_CLR[self.clr_idx][1], annot_mode=True)

        self.draw_thumbnail()
        self.update_loadbutton()

    def toggle_color(self, clr_idx):
        self.platform.change_color(self.BTN_CLR[self.clr_idx][1])
        self.window['-BTNTGL-'].update(button_color=self.BTN_CLR[self.clr_idx])
        self.draw_thumbnail()

    def change_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.MODE)

        self.window['-BTNMOD-'].update(PW.MODE[self.mode_idx])
        self.window['-BTNTGL-'].update(PW.BTNTGL_TXT[self.mode_idx])

        if self.mode_idx:  # = 1
            self.window['-BTNTGL-'].update(button_color=self.BTN_CLR[self.clr_idx])
        else:  # = 0
            self.window['-BTNTGL-'].update(button_color=sg.DEFAULT_BUTTON_COLOR)

        self.window['-BTNFINC-'].update(PW.BTNFIN_TXTC[self.mode_idx])
        self.window['-BTNFINA-'].update(PW.BTNFIN_TXTA[self.mode_idx], disabled=PW.BTNFIN_DISABLE[self.mode_idx])

        self.platform.switch_mode(self.mode_idx, self.BTN_CLR[self.clr_idx][1])
        self.draw_thumbnail()
        self.update_loadbutton()

    def imagelist_window(self):
        imagenames = list(self.imagepaths.keys())
        btn_clrs = []
        for wsi_name in imagenames:
            jpg_exist = os.path.exists(self.imagepaths[wsi_name].replace('_o.jpg', '_i.jpg'))
            annot_exist = self.check_annotationfile(wsi_name, annot_name=PG.annotation_filename)
            guide_exist = self.check_annotationfile(wsi_name, annot_name=PG.generation_filename)
            patch_inuse_exist = wsi_name in self.df_inuse()['Case'].values
            patch_indisuse_exist = wsi_name in self.df_indisuse()['Case'].values
            if patch_inuse_exist:
                btn_clr = ('white', 'blue')
            elif patch_indisuse_exist:
                btn_clr = ('white', 'red')
            elif (jpg_exist or annot_exist) and guide_exist:
                btn_clr = ('white', 'lime')
            elif annot_exist:
                btn_clr = ('black', 'yellow')
            else:
                btn_clr = sg.DEFAULT_BUTTON_COLOR
            btn_clrs.append(btn_clr)
        msg = 'Blue, patch already created. Red, patch tempolarily disused (usage=False) exist. Green,' \
              ' Ready to generate patches. Yellow, Jpg conversion of annotation required. DarkBlue, annotation required.'
        title = 'List of image files'

        lw = ListWin(imagenames, btn_clr=btn_clrs, msg=msg, title=title)
        selected_wsiname = lw.start()
        if selected_wsiname != self.platform.wsi_name and selected_wsiname != None:  # do nothing when select the same image
            self.renew_image(img_name=selected_wsiname, mode_idx=self.mode_idx)
            self.update_message()
            self.draw_thumbnail()
            self.update_imagename()
            self.image_counter = list(self.imagepaths.keys()).index(selected_wsiname)

    def show_inset(self, x, y, magnification):
        # (x, y) clicked location on thumbnail window
        # magnification: for inset magnification. x1: original 1 pixel = inset 1 pixel, x2: original 2 pixels = inset 1 pixel. Like that.

        def change_magnification(event):  # function of wheel
            nonlocal magnification, g, win, edgex, edgey, x, y, ixy
            delta = event.delta // 120
            _mag = [int(i.replace('x', '')) for i in self.MAGNIFICATION]
            mag_idx = min(max(0, _mag.index(magnification) + delta), len(_mag) - 1)
            magnification = _mag[mag_idx]
            edgex, edgey = calculate_edge(magnification)
            ixy = inset_xy(self.platform.RX(x), self.platform.RY(y))
            draw_inset(x, y)

        def insetkey_zoom(delta):
            nonlocal magnification, g, win, edgex, edgey, x, y, ixy
            _mag = [int(i.replace('x', '')) for i in self.MAGNIFICATION]
            mag_idx = min(max(0, _mag.index(magnification) + delta), len(_mag) - 1)
            magnification = _mag[mag_idx]
            edgex, edgey = calculate_edge(magnification)
            ixy = inset_xy(self.platform.RX(x), self.platform.RY(y))
            draw_inset(x, y)

        def calculate_edge(magnification):
            return self.INSET_SIZE[0] * magnification, self.INSET_SIZE[1] * magnification

        def inset_xy(x, y):  # x, y of thumbnail window  (center) is converted to inset area of real x, y
            nonlocal edgex, edgey
            x0 = min(max(0, x - edgex // 2), self.platform.imgsize[0] - edgex)
            x1 = x0 + edgex - 1
            y0 = min(max(0, y - edgey // 2), self.platform.imgsize[1] - edgey)
            y1 = y0 + edgey - 1
            return (x0, y0, x1, y1)

        def crop_img(x, y):
            _x = self.platform.RX(x)
            _y = self.platform.RY(y)
            crop_img = self.platform.img.crop(inset_xy(_x, _y))
            inset_img = crop_img.resize(self.INSET_SIZE, Image.ANTIALIAS)
            bio = io.BytesIO()
            inset_img.save(bio, format="PNG")
            inset_bio = bio.getvalue()
            return inset_bio

        def centralize(event):  # function of shift-click
            nonlocal g, x, y, ixy
            dx = (event.x - self.INSET_SIZE[0] // 2) * magnification
            dy = (event.y - self.INSET_SIZE[1] // 2) * magnification
            x += dx // self.platform.magnification
            y -= dy // self.platform.magnification
            ixy = inset_xy(self.platform.RX(x), self.platform.RY(y))
            draw_inset(x, y)

        def delete_point(event):
            nonlocal x, y
            if len(self.platform.xlist) > 0:
                del self.platform.xlist[-1]
                del self.platform.ylist[-1]
                self.platform.xy[-1] = [self.platform.xlist, self.platform.ylist]
                draw_inset(x, y)

        def draw_inset(x, y):
            # normal line is omitted in inset
            nonlocal g, ixy
            default_dotclr = 'blue'
            # parameters for formatting spline curve
            splinedot_max = 500
            dc = [[default_dotclr] * len(self.platform.clist), self.platform.clist,
                  [default_dotclr] * len(self.platform.clist)]  # dot color
            ds = [10, 2, 10]  # dot size
            lw = [1, 5, 1]  # line width

            dot_color = dc[self.mode_idx]
            dot_size = ds[self.mode_idx]
            line_color = self.platform.clist
            line_width = lw[self.mode_idx]

            def InX(rx):  # convert real x to inset x
                nonlocal ixy
                inset_x = int((rx - ixy[0]) / (ixy[2] - ixy[0]) * self.INSET_SIZE[0])
                return inset_x

            def InY(ry):
                nonlocal ixy
                inset_y = int((ixy[3] - ry) / (ixy[3] - ixy[1]) * self.INSET_SIZE[1])
                return inset_y

            def _draw_spline(xlist, ylist, line_cnt, dc=dot_color, lc=line_color):
                nonlocal g, ixy
                spline_pointfactor = 100  # magic number. Change the number of normal lines (if larger, then fewer)
                for x, y in zip(xlist, ylist):
                    g.DrawPoint((InX(x), InY(y)), size=dot_size, color=dc)
                if len(xlist) < 2:
                    return
                elif len(xlist) == 2:  # points must be more than 3 for spline
                    xl, yl = xlist, ylist
                else:
                    # calculate appropriate number of points
                    xyn = np.array([xlist, ylist]).T
                    dif = np.diff(xyn, axis=0)
                    length = np.sum(np.linalg.norm(dif, axis=-1))
                    points = int(min(max(5, length // spline_pointfactor), splinedot_max))  #

                    xl, yl = PG.spline(xlist, ylist, point=points)
                before_value = (0, 0)
                cnt = 0

                for x, y in zip(xl, yl):

                    if cnt != 0:
                        g.DrawLine(before_value, (InX(x), InY(y)), color=lc, width=line_width)

                    before_value = (InX(x), InY(y))
                    cnt += 1

            ### Main of def draw_inset ###
            g.Erase()
            inset_bio = crop_img(x, y)
            g.DrawImage(data=inset_bio, location=(0, self.INSET_SIZE[1]))

            for line_cnt, (xy, dclr, lclr) in enumerate(zip(self.platform.xy, dot_color, line_color)):
                _draw_spline(xy[0], xy[1], line_cnt, dc=dclr, lc=lclr)

        ### Main of show_inset ###
        # clicked out of the thumbnail
        if x >= self.platform.thumbnail_size[0] or y >= self.platform.thumbnail_size[1]:
            return

        # calculate inset window location
        xi = min(max(0, x - self.INSET_SIZE[0] // 2), self.WINDOW_SIZE[0] - self.INSET_SIZE[0])
        if y - self.platform.thumbnail_size[1] // 2 < 0:
            yi = 0
        else:
            yi = self.WINDOW_SIZE[1] - self.INSET_SIZE[1]

        edgex, edgey = calculate_edge(magnification)  # edgex, edgey: real width and height for the INSET image
        ixy = inset_xy(self.platform.RX(x), self.platform.RY(y))  # area of inset by real x, y.

        g = sg.Graph(self.INSET_SIZE, (0, 0), self.INSET_SIZE, enable_events=True,
                     drag_submits=True, key='-IMG-')
        layout = [[g]]
        win = sg.Window(
            '(Click or drag) Point. (Shift-click) Center. (> or Wheel) Zoom in (< or Wheel) Zoom out. (z) Delete point. (v) Change line. (Esc) Close',
            layout, size=self.INSET_SIZE, location=(xi, yi),
            return_keyboard_events=True, keep_on_top=True, modal=True)

        # Forced to close the inset window by keep_on_top
        win.Finalize()
        win['-IMG-'].Widget.bind('<MouseWheel>', change_magnification)
        win['-IMG-'].Widget.bind('<Shift-1>', centralize)

        draw_inset(x, y)
        while True:
            event, values = win.read()
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
            if event == 'Escape:27':  # why this value? Escape key
                break
            if event == '.':
                insetkey_zoom(1)
            if event == ',':
                insetkey_zoom(-1)
            if event == '-IMG-':
                _x = values['-IMG-'][0]
                _y = values['-IMG-'][1]
                rx = ixy[0] + (ixy[2] - ixy[0]) * _x // self.INSET_SIZE[0]
                ry = ixy[1] + (ixy[3] - ixy[1]) * (self.INSET_SIZE[1] - _y) // self.INSET_SIZE[1]

                # #platformにupdate
                self.platform.update(self.platform.IX(rx), self.platform.IY(ry))
                draw_inset(x, y)
                self.draw_thumbnail()
            if event == 'z':
                _ = self.platform.delete()
                draw_inset(x, y)
                self.draw_thumbnail()
            if event == 'v':
                _ = self.platform.next_line(self.BTN_CLR[self.clr_idx][1])
                draw_inset(x, y)
                self.draw_thumbnail()
            if event == 'c' and self.mode_idx == 0:
                self.platform.clkwise[-1] = not self.platform.clkwise[-1]
                draw_inset(x, y)
                self.draw_thumbnail()
            if event == 'c' and self.mode_idx == 1:
                self.clr_idx = (self.clr_idx + 1) % len(self.BTN_CLR)
                self.toggle_color(self.clr_idx)
                draw_inset(x, y)
                self.draw_thumbnail()

        win.close()
        # At the inset close, thumbnail is renewed
        self.draw_thumbnail()

    def save_annotatedimage(self):
        if os.path.exists(self.platform.imgpath.replace('_o.jpg', '_i.jpg')):
            r = sg.PopupYesNo('Annotated file already exists. Overwrite?',
                              no_titlebar=True)
            if r != 'Yes':
                return 'Annotation has NOT been converted to jpg.'

        self.platform.draw_annotation()
        return 'Saved as {} in {}'.format(os.path.basename(self.platform.imgpath).replace('_o', '_i'),
                                          self.IMAGEFOLDER)

    def check_annotationfile(self, wsi_name, annot_name=None):
        if annot_name == None:
            annot_name = self.platform.annot_name
        return os.path.isfile(self.ANNOTATIONFOLDER + os.sep + annot_name + wsi_name + '.pkl')

    def default_view(self):
        mag = self.platform.max_magnification
        self.platform.change_magnification(mag)
        self.draw_thumbnail()

    def run_AI(self, wsi_name, patchnum):
        ini_filename = 'ini.pkl'
        init = IniHandler(ini_filename)

        # make default initialization folder
        _modelpath = ''
        if len(glob.glob(self.AIFOLDER + os.sep + '*.h5')) > 0:
            _modelpath = glob.glob(self.AIFOLDER + os.sep + '*.h5')[0]
        default_ini = {'display_mode_rai': 'Weight', 'tissue_number_rai': '2', 'output_folder_rai': PW.HOMEDIRECTORY}
        default_ini['AI_rai'] = _modelpath
        init.load_ini(default_ini)  # if ini file exists, load, else default_ini

        self.platform.calculate_patchlocation(patchnum)
        patchlocation = self.platform.location
        patchset = self.platform.generate_imageset()

        # window layout parameters
        s0 = (12, 1)
        s1 = (60, 1)
        display_modes = {'-BTNW-': 'Weight', '-BTNM-': 'MaxClass', '-BTNE-': 'Each',
                         '-BTNC-': 'CustomWeight'}  # Still not used
        tissue_numbers = ('1', '2', '3', '4', '5', '6', '7', '8')  # To do: Autocount the number of fragments
        run_once = False  # if run once, there must be a predictions in analyzer

        btn_layout = [sg.B(list(display_modes.values())[i], size=s0, disabled=not run_once,
                           key=list(display_modes.keys())[i]) for i in range(len(display_modes))]
        btn_layout.insert(0, sg.T('Display', size=s0))

        layout = [
            [sg.T('AI', size=s0), sg.InputText(default_text=init.ini['AI_rai'], size=s1, key='-AI-'), sg.FileBrowse()],
            [sg.T('Tissue number', size=s0),
             sg.Spin(values=tissue_numbers, initial_value=init.ini['tissue_number_rai'], size=s0, key='-TN-'),
             sg.Checkbox('Vertical', size=s0, key='-CHB-'), ],
            [sg.T('Output folder', size=s0), sg.InputText(default_text=init.ini['output_folder_rai'],
                                                          size=s1, key='-OF-'),
             sg.FolderBrowse()],
            [sg.T('Analyze', size=s0), sg.B('Run', size=s0)],
            btn_layout,
            [sg.B('Close', size=s0)]
            ]
        subwin = sg.Window('Run AI', layout, keep_on_top=True, modal=False, finalize=True)

        while True:
            event, values = subwin.Read()
            if event in (sg.WIN_CLOSED, 'Close'):
                heatmap_set = None
                break
            if event == 'Run':
                sg.PopupQuickMessage('Analyzing..', keep_on_top=True, background_color='yellow', text_color='blue')
                init.update_ini('AI_rai', values['-AI-'])
                init.update_ini('tissue_number_rai', int(values['-TN-']))
                init.update_ini('output_folder_rai', values['-OF-'])

                analyzer = AP(wsi_name=wsi_name,
                              patch_set=patchset,
                              patch_location=patchlocation,
                              modelpath=init.ini['AI_rai'],
                              cluster_num=init.ini['tissue_number_rai'])
                # check patch_set, model presence and image size
                (flag, msg) = analyzer.check_validity()
                if flag:
                    analyzer.calculate_prediction()
                    init.save_ini()
                    self.update_message('Prediction done.')
                    run_once = True
                    for k in display_modes.keys():
                        subwin[k].update(disabled=not run_once)

                else:
                    self.update_message(msg, color='red')

            if event in (display_modes.keys()):
                weights = [[]]

                if display_modes[event] == 'CustomWeight':
                    weights = self.CUSTOM_WEIGHTS
                elif display_modes[event] == 'Each':
                    weights = [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0],
                               [0, 0, 0, 3]]  # [1, 0, 0, 0] etc give identical results
                elif display_modes[event] == 'MaxClass':
                    weights = [[0, 1, 2, 3]]
                else:  # 'Weights':
                    weights = [[0, 0, 3, 0], [0, 0, 0, 3]]

                hm = analyzer.draw_heatmap(display_mode=display_modes[event],
                                           vertical=values['-CHB-'],
                                           default_filename='Heatmap',
                                           weights=weights)
                self.display_heatmap(hm, values['-OF-'], display_modes[event])
        subwin.close()
        return heatmap_set

    def display_heatmap(self, heatmap_set, output_folder, display_mode):
        default_filename = 'Heatmap'

        def calculate_adjustimgsize(hx, hy, wx, wy):  # heatmapx, heatmapy, windowx, windowy
            magnification = [wx / hx, wy / hy]
            if magnification.index(min(magnification)):  # wy/hy is smaller, thus height is limit factor
                new_size = (wy // hy * hx, wy // hy * hy)
            else:
                new_size = (wx // hx * hx, wx // hx * hy)
            return new_size

        def resize_img(img, size):
            img = img.resize(size)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            img_bio = bio.getvalue()
            return img_bio

        margin = 10
        if heatmap_set == None or len(heatmap_set) == 0:
            return
        (x, y) = heatmap_set[0].size
        img_panel = Image.new('RGB', (x + margin * 2, (y + margin) * len(heatmap_set) + margin), (0, 0, 0))
        for i, im in enumerate(heatmap_set):
            img_panel.paste(im, (margin, margin + (y + margin) * i))

        win_size = calculate_adjustimgsize(img_panel.size[0], img_panel.size[1],
                                           self.INSET_SIZE[0], self.INSET_SIZE[1])

        g = sg.Graph(win_size, (0, 0), win_size, enable_events=False)

        img_bio = resize_img(img_panel, win_size)

        # window layout parameters for heatmap display
        s0 = (10, 1)
        btn_space = 50

        layout = [[g],
                  [sg.B('Save', size=s0), sg.B('Close', size=s0)]]
        _title = 'Heatmap {}'.format(list(self.imagepaths.keys())[self.image_counter])
        subwin = sg.Window(_title, layout, size=(win_size[0], win_size[1] + btn_space), keep_on_top=True)
        subwin.Finalize()
        g.DrawImage(data=img_bio, location=(0, win_size[1]))
        while True:
            event, values = subwin.read()
            if event in (sg.WIN_CLOSED, 'Close'):
                break
            if event == 'Save':
                savepath = output_folder + os.sep + default_filename + self.platform.wsi_name + '_' + display_mode + '.png'
                img_panel.save(savepath)
                break
        subwin.close()

    def read_imagefolder(self, image_folder):
        def read(image_folder):
            _imagepaths = glob.glob(image_folder + os.sep + '*_o.jpg')
            imagepaths = OrderedDict()
            for ip in _imagepaths:
                imagepaths[os.path.basename(ip).replace('_o.jpg', '')] = ip
            return imagepaths

        _impaths = glob.glob(image_folder + os.sep + '*_o.jpg')
        _input = image_folder
        while _impaths == []:
            _input = sg.popup_get_folder('No image is available. Choose a folder that contains *_o.jpg files.',
                                         size=(80, 1), keep_on_top=True)

            if _input == None:
                sys.exit()
            else:
                _impaths = glob.glob(_input + os.sep + '*_o.jpg')

        image_folder = _input
        image_folder = image_folder.replace('/', os.sep)

        imagepaths = read(image_folder)
        img_counter = 0

        return image_folder, imagepaths, img_counter

    def disuse_patch(self):
        casein_df = list(set(self.df_inuse().Case))
        casein_df.sort()
        if len(casein_df) == 0:
            self.update_message('No patch exists.', color='red')
            return
        lw = ListWin(casein_df, msg='Select one to disuse.',
                     title='List of cases in the patch collection')
        selected_case = lw.start()

        self.df.loc[self.df.Case == selected_case, 'usage'] = False
        msg = self.save_labelfile()
        self.update_message(msg)

    def restore_patch(self):
        disusedcasein_df = list(set(self.df_indisuse().Case))
        disusedcasein_df.sort()
        if len(disusedcasein_df) == 0:
            self.update_message('No restorable patch exists.', color='red')
            return
        lw = ListWin(disusedcasein_df, msg='Select one that you want to restore.',
                     title='List of cases that have disused patches')
        selected_case = lw.start()
        self.df.loc[self.df.Case == selected_case, 'usage'] = True
        msg = 'Patches of {} are restored'.format(selected_case)
        _msg = self.save_labelfile()
        self.update_message(PW.add_msg(msg, _msg))

    def save_labelfile(self, location=None, name='Label'):
        if location == None:
            location = self.PATCHFOLDER
        labelfilename = name + time_stamp() + '.pkl'
        self.df.to_pickle(location + os.sep + labelfilename)
        msg = 'The label has been saved as {} in {}.'.format(labelfilename, location)
        return msg

    def load_labelfile(self, location=None, name='Label'):
        if location == None:
            location = self.PATCHFOLDER

        labelpath = glob.glob(location + os.sep + '*.pkl')

        if len(labelpath) == 1:
            df = pd.read_pickle(labelpath[0])
            msg = 'Label file ({}) loaded.'.format(os.path.basename(labelpath[0]))
        elif len(labelpath) == 0:
            df = pd.DataFrame(index=[], columns=PW.COLUMNS)
            msg = 'A new label has been generated (Not saved yet).'
        else:
            df = pd.read_pickle(labelpath[-1])
            _msg = 'Warning! Since multiple labels exist in the assigned folder,' \
                   'the presumably latest label {} is loaded'.format(os.path.basename(labelpath[-1]))
            msg = [_msg, 'red']
        return df, msg

    def convertannot_tojpg(self, avoid_overwrite=True):
        annot_paths = glob.glob(self.ANNOTATIONFOLDER + os.sep + PG.annotation_filename + '*.pkl')
        cnt = 0
        for annot_path in annot_paths:
            sg.OneLineProgressMeter('Processing', cnt + 1, len(annot_paths), orientation='h')
            case = os.path.basename(annot_path).replace(PG.annotation_filename, '').replace('.pkl', '')
            im_path = self.imagepaths[case]

            if os.path.exists(im_path.replace('_o.jpg', '_i.jpg')) and avoid_overwrite:
                continue

            _platform = PG(im_path, None, None, self.PATCHFOLDER, self.WINDOW_SIZE,
                           initialbutton_color=self.BTN_CLR[self.clr_idx][1], annot_mode=True)
            _platform.load_annotation(self.ANNOTATIONFOLDER)
            _platform.draw_annotation()
            cnt += 1

        msg = '{} _i.jpg files were created in {}.'.format(cnt, self.IMAGEFOLDER)
        if cnt == 0:
            msg = 'No jpg file was created.'
        return msg

    @staticmethod
    def add_msg(msg1, msg2):
        return msg1 + ' ' + msg2


#############MAIN#############
Image.MAX_IMAGE_PIXELS = 5000000000

pw = PW()
