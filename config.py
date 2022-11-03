# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#

__C.DATASETS = edict()
__C.DATASETS.HEARTSEG = edict()
__C.DATASETS.HEARTSEG.IMG_ROOT = '/home/ds17/Documents/phd/p2vse_a/datasets/heart_seg/heart_render_binary_2_4'
__C.DATASETS.HEARTSEG.TAXONOMY_FILE_PATH = '/home/ds17/Documents/phd/p2vse_a/datasets/HeartSeg.json'
__C.DATASETS.HEARTSEG.RENDERING_PATH = os.path.join(__C.DATASETS.HEARTSEG.IMG_ROOT, '%s/%s/*.png')
__C.DATASETS.HEARTSEG.VOXEL_PATH = '/home/ds17/Documents/phd/p2vse_a/datasets/heart_seg/voxel_vtk_128/%s/%s/model.npy'

__C.DATASETS.ULTRASOUND = edict()
__C.DATASETS.ULTRASOUND.RENDERING_PATH = os.path.join('/home/ds17/Documents/phd/switcheroo/camus_dia/%s/%s/*.png')


__C.DATASET = edict()
__C.DATASET.MEAN = [23.94071942, 23.94071942, 23.94071942]
__C.DATASET.STD = [1.6628881, 1.6628881, 1.6628881]
__C.DATASET.TRAIN_DATASET = 'HeartSeg'
__C.DATASET.TEST_DATASET = 'HeartSeg'
__C.DATASET.INFERENCE_DATASET = 'UltraSound'

__C.LABELS = edict()
__C.LABELS.METADATA = edict()
__C.LABELS.LV = 1
__C.LABELS.RV = 2
__C.LABELS.LA = 3
__C.LABELS.RA = 4
__C.LABELS.AORTA = 5
__C.LABELS.MITRAL_VALVE = 7
__C.LABELS.AORTIC_VALVE = 9
__C.LABELS.PULMONARY_VALVE = 10
__C.LABELS.METADATA.LABEL_NAME = 'elemTag'
__C.LABELS.METADATA.VOXEL_SCALING = 1.2685719579160966


# Common
#
__C.CONST = edict()
__C.CONST.DEVICE = '0'
__C.CONST.RNG_SEED = 1
__C.CONST.LABEL_WEIGHTS = [1.00000000e+00, 2.32134388e+01, 6.14222964e+01, 1.32491570e+02, 1.65160308e+02,
                           1.53597106e+03, 9.75509972e+02, 2.01958805e+03, 1.73501937e+03]
__C.CONST.IMG_W = 224
__C.CONST.IMG_H = 224
__C.CONST.BATCH_SIZE = 4
__C.CONST.N_VIEWS_RENDERING = 2
__C.CONST.NUM_WORKER = 10
__C.CONST.TEST_SAVE_NUMBER = 125
__C.CONST.BINARIZE_VOLUME_LABELS = False
__C.CONST.SAVE_MONTAGE = True
__C.CONST.MIN_SAVE_VTK_MESHES_EPOCH = 20
__C.CONST.SAVE_VTK_MESHES = False
__C.CONST.SAVE_MESH_PLOTS = True
__C.CONST.MULTI_LABEL = True
__C.CONST.NUM_CLASSES = 9

# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = os.path.join(os.getcwd(), 'output')
__C.DIR.IOU_SAVE_PATH = os.path.join(__C.DIR.OUT_PATH, 'iou_scores.xlsx')
__C.DIR.RANDOM_BG_PATH = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.LEAKY_VALUE = .2
__C.NETWORK.TCONV_USE_BIAS = False
__C.NETWORK.LOSS_FUNC = 'crossentropyloss'
__C.NETWORK.EMBED_DIMS = [256, 512, 512]
__C.NETWORK.DROP_RATE = 0.
__C.NETWORK.DROP_PATH_RATE = 0.
__C.NETWORK.DEPTHS = [1, 1, 1]


__C.NETWORK.DEPTH = 6
__C.NETWORK.DIM = 8192
__C.NETWORK.KERNEL_SIZE = 5
__C.NETWORK.PATCH_SIZE = 7


#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.TRAIN_TIME_INFERENCE = False
__C.TRAIN.NUM_EPOCHS = 1000
__C.TRAIN.LR = 1e-3
__C.TRAIN.MILESTONES = [10, 20, 50]
__C.TRAIN.MIN_LR = 1e-6
__C.TRAIN.MULTI_IMG = True
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY = 'adam'  # available options: sgd, adam
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.MOMENTUM = .9
__C.TRAIN.GAMMA = .5
__C.TRAIN.SAVE_FREQ = 1050  # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING = False

#
# Testing options
#
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.TEST_NETWORK = False
__C.TEST.VOL_OR_RENDER_SAVE = 'volume'

