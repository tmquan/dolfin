#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan <quantm@unist.ac.kr>

###############################################################################
# Uitilities packages
import os, sys
import natsort
import argparse
import glob
from six.moves import map, zip, range
import numpy as np

###############################################################################
# Tensor packages
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
import tensorflow as tf

###############################################################################
# Image packages
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation

###############################################################################
# Meta configurations
SEED  = 2017
DIMN  = 4       # Epoch size
DIMB  = 3       # Batch size

DIMZ  = 1       # Number of slices in volume, =1 for 2D image
DIMY  = 256     # Height
DIMX  = 256     # Width
DIMC  = 3       # Channel, =3 for color images

dictSize  = 24; # Number of atoms in the dictionary
dataSize  = 6;  # Number of images in the entire dataset

atomShape = [   1, dictSize, DIMC,    1,   32,   32]
dataShape = [DIMB,        1, DIMC, DIMZ, DIMY, DIMX]
iterShape = [   1, dictSize, DIMC, DIMZ, DIMY, DIMX] # For padding
blobShape = [DIMB, dictSize, DIMC, DIMZ, DIMY, DIMX]

# Define a bunch of parameters, if we run on the graph base, needs to wrap by a class of parameters
LAMBDA = 0.1 # Sparsity constraint
RHO    = 0.001
SIGMA  = 0.001

###############################################################################
class DolfinModelDesc(ModelDesc):
    def collect_variables(self, d_scope='atom', x_scope='code'):   
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)
        # assert self.d_vars
        self.x_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, x_scope)
        # assert self.x_vars
###############################################################################
def bndcrop(d, new_shape=atomShape, name='bndcrop'):
    new_shape = [None, new_shape[1], new_shape[2], new_shape[3], new_shape[4], new_shape[5]]
    new_d = d[:,:new_shape[1], :new_shape[2], :new_shape[3], :new_shape[4], :new_shape[4]]
    return new_d

def zeropad(d, new_shape=iterShape, name='zeropad'):
    old_shape = d.get_shape().as_list()
    new_shape = [None, new_shape[1], new_shape[2], new_shape[3], new_shape[4], new_shape[5]]
    new_d = tf.pad(d, [
                        [0, 0], # dimb
                        [0, 0], # dimk
                        [0, 0], # dimc
                        [0, new_shape[3]-old_shape[3]], # dimz
                        [0, new_shape[4]-old_shape[4]], # dimy
                        [0, new_shape[5]-old_shape[5]], # dimx

                      ])
    return new_d


def custom_padding(s, dataShape, atomShape, name='broadcasting'):
    # print s.get_shape().as_list()
    print s
    # print s.get_shape().as_list()
    # assert thisShape == dataShape
    dictSize = atomShape[1]
    with tf.variable_scope(name):
        paddings = [[0, 0],            # dimb
                    [0, dictSize-1],   # dimk
                    [0, 0],            # dimc
                    [0, 0],            # dimz 
                    [0, 0],            # dimy 
                    [0, 0]             # dimx
                    ]
        S = tf.pad(s, paddings, "SYMMETRIC")
        print S.get_shape().as_list()
        return S

def filter_length(np_list):
    desire_list = [(x-1)*2 for x in np_list[-3:]]
    return tf.cast(tf.stack(desire_list), tf.int32)
###############################################################################
class DolfinModel(DolfinModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 1, DIMZ, DIMY, DIMX, DIMC), 'inputS'), 
                InputDesc(tf.float32, (None, DIMY, DIMX, DIMC), 's_tmp')
                ]



    def _init_variable(self, 
        D_init=None, 
        X_init=None):
        if D_init is None:
            D_init = tf.contrib.layers.variance_scaling_initializer()
        if X_init is None:
            X_init = tf.contrib.layers.variance_scaling_initializer()
        return D_init, X_init



    def _build_graph(self, inputs):
        s, s_tmp = inputs

        s = (s / 128.0 - 1.0)
        s_tmp = (s_tmp / 128.0 - 1.0)
        # Tranpose the s for fourier transform
        # from b, k, z, y, x, c, to b, k, c, z, y, x
        # from 012345 to 015234
        s = tf.transpose(s, [0,1,5,2,3,4])
        ######################################################################
        # Construct the blob size from data size
        # This expand the input s along dimension k of dictionary
        with tf.variable_scope('initialization'):
            S  = custom_padding(s, dataShape, atomShape)
            S  = tf.identity(S, name='S')
            Sc = tf.cast(S, tf.complex64, name='Sc') # Complex type of S
            Sf = tf.spectral.fft3d(Sc, name='Sf')
            Sf = tf.identity(Sf, name='Sf')

            # Gets an existing variable with these parameters or create a new one.
            # D_init, X_init = self._init_variable()
            D_init = tf.contrib.layers.variance_scaling_initializer()
            G_init = tf.contrib.layers.variance_scaling_initializer()
            X_init = tf.contrib.layers.variance_scaling_initializer()
            Y_init = tf.contrib.layers.variance_scaling_initializer()
            U_init = tf.contrib.layers.variance_scaling_initializer()
            H_init = tf.contrib.layers.variance_scaling_initializer()
            
        
            D  = tf.get_variable('D', atomShape, initializer=D_init)
            D  = bndcrop(D, new_shape=atomShape, name='D_bndcrop')
            D  = zeropad(D, new_shape=iterShape, name='D_zeropad')
            Dc = tf.cast(D, tf.complex64, name='Dc')
            Df = tf.spectral.fft3d(Dc, name='Df')
            Df = tf.identity(Df, name='Df')

            G  = tf.get_variable('G', iterShape, initializer=G_init)
            Gc = tf.cast(G, tf.complex64, name='Gc')
            Gf = tf.spectral.fft3d(Gc, name='Gf')
            Gf = tf.identity(Gf, name='Gf')

            H  = tf.get_variable('H', iterShape, initializer=H_init)
            Hc = tf.cast(H, tf.complex64, name='Hc')
            Hf = tf.spectral.fft3d(Hc, name='Hf')
            Hf = tf.identity(Hf, name='Hf')




            X  = tf.get_variable('X', blobShape, initializer=X_init)
            Xc = tf.cast(X, tf.complex64, name='Xc')
            Xf = tf.spectral.fft3d(Xc, name='Xf')
            Xf = tf.identity(Xf, name='Xf')

            Y  = tf.get_variable('Y', blobShape, initializer=Y_init)
            Yc = tf.cast(Y, tf.complex64, name='Yc')
            Yf = tf.spectral.fft3d(Yc, name='Yf')
            Yf = tf.identity(Yf, name='Yf')

            U  = tf.get_variable('U', blobShape, initializer=U_init)
            Uc = tf.cast(U, tf.complex64, name='Uc')
            Uf = tf.spectral.fft3d(Uc, name='Uf')
            Uf = tf.identity(Uf, name='Uf')


        with tf.variable_scope('atom'):
            GSf = tf.multiply(tf.conj(Gf, name='GfT'), Sf, name='GSf')
        with tf.variable_scope('code'):
            YSf = tf.multiply(tf.conj(Yf, name='YfT'), Sf, name='YSf')


        with tf.variable_scope('atom'):
            s_out1 = Conv2D('conv0', s_tmp, out_channel=3, kernel_shape=1, padding='SAME')
            self.d_loss = tf.reduce_mean(tf.abs(s_out1-s_tmp)) + tf.reduce_mean(S)

        with tf.variable_scope('code'):
            s_out2 = Conv2D('conv1', s_tmp, out_channel=3, kernel_shape=1, padding='SAME')
            self.x_loss = tf.reduce_mean(tf.abs(s_out2-s_tmp)) + tf.reduce_mean(S)
        ######################################################################
        # Loss
        self.collect_variables('atom', 'code')
        ######################################################################


    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


###############################################################################
class DolfinTrainer(FeedfreeTrainerBase):
    def __init__(self, config):
        self._input_source = QueueInput(config.dataflow)
        super(DolfinTrainer, self).__init__(config)

    def _setup(self):
        super(DolfinTrainer, self)._setup()
        self.build_train_tower()
        opt = self.model.get_optimizer()

        # self.d_min = opt.minimize(self.model.d_loss, var_list=self.model.d_vars, name='d_op')
        # self.train_op = self.d_min
        # by default, run one d_min after one g_min
        self.d_min = opt.minimize(self.model.d_loss, var_list=self.model.d_vars, name='d_op')
        with tf.control_dependencies([self.d_min]):
            self.x_min = opt.minimize(self.model.x_loss, var_list=self.model.x_vars, name='x_op')
        self.train_op = self.x_min

###############################################################################
class ImageDataFlow(RNGDataFlow):
    def __init__(self, imageDir, size, dtype='float32', is_training=True):
        self.dtype  = dtype
        self.imageDir = imageDir
        self._size  = size
        self.is_training = is_training

    def size(self):
        return self._size

    def reset_state(self):
        self.rng = get_rng(self)   
        pass

    def get_data(self, shuffle=True):
        # self.reset_state()
        images = glob.glob(self.imageDir + '/*.png')

        if self._size==None:
            self._size = len(images)
        from natsort import natsorted
        images = natsorted(images)
        print images
        for k in range(self._size):
            from random import randrange
            rand_index = randrange(0, len(images))
            image = skimage.io.imread(images[rand_index])
            image = skimage.transform.resize(image, output_shape=(DIMY, DIMX), order=0)
            image = skimage.img_as_ubyte(image)
            image = np.squeeze(image)
            print image.shape
            yield [image.astype(np.uint8), image.astype(np.uint8)]
###############################################################################
def get_data(dataDir, isTrain=True):
    if isTrain:
        augs = []
        num = 6
    else:
        augs = []
        num = 1

   
    df = ImageDataFlow(dataDir, size=DIMN, is_training=isTrain)
    df = BatchData(df, DIMB)
    # df = PrefetchDataZMQ(df, 2 if isTrain else 1)
    return df
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        help='comma separated list of GPU(s) to use.')
    parser.add_argument('--dataDir',    help='the image directory.', 
                                        required=True)
    parser.add_argument('--load',       help='load pretrained model')
    parser.add_argument('--sample',     help='run sampling one instance', 
                                        action='store_true')
    args = parser.parse_args()

    # Enable logger
    logger.auto_set_dir()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    dataflow = get_data(args.dataDir)
    PrintData(dataflow, num=2)
    if args.sample:
        sample(args.load)
    else:
        config = TrainConfig(
            model=DolfinModel(),
            dataflow=dataflow,
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=5),
                ScheduledHyperParamSetter(
                    'learning_rate',
                    [(100, 2e-4), (200, 0)], interp='linear'),
            ],
            max_epoch=195,
            session_init=SaverRestore(args.load) if args.load else None
        )

        DolfinTrainer(config).train()