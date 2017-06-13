#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan <quantm@unist.ac.kr>

###############################################################################
# Uitilities packages
import os, sys
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

DIMN  = 6		# Epoch size
DIMB  = 4 		# Batch size
DIMZ  = 1		# Number of slices in volume, =1 for 2D image
DIMY  = 512		# Height
DIMX  = 512		# Width
DIMC  = 1		# Channel



###############################################################################
class DolfinModel(ModelDesc):
	def _get_inputs(self):
		pass

	def _build_graph(self, inputs):
		pass

	def _get_optimizer(self):
		pass

###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', 		help='the image directory.', 
    									required=True)
    parser.add_argument('--load', 		help='load pretrained model')
    parser.add_argument('--sample',     help='run sampling one instance', 
                                        action='store_true')
    args = parser.parse_args()



     if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu