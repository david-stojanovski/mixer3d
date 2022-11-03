# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import logging
import os
import random
from datetime import datetime as dt
from time import time

import torch
import torch.backends.cudnn
import torch.utils.data
from tensorboardX import SummaryWriter

import utils.data_loaders
import utils.data_transforms
import utils.helpers
from core.test import test_net
from utils.average_meter import AverageMeter
from models.archs import ConvMixer, Decoder
from torch.optim import lr_scheduler


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        # utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        # utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        # utils.data_transforms.RandomFlip(),
        # utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor()
    ])
    val_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor()
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
        batch_size=1,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    # Set up networks
    model = ConvMixer(cfg)
    model_decoder = Decoder(cfg)

    model.apply(utils.helpers.init_linear)
    model_decoder.apply(utils.helpers.init_weights)

    # logging.debug('Parameters in model: %d.' % (utils.helpers.count_parameters(model)))
    # print('Parameters in model: %d.' % (utils.helpers.count_parameters(model)))

    # Initialize weights of networks
    # model.apply(utils.helpers.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        model_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=cfg.TRAIN.LR)
        model_decoder_solver = torch.optim.Adam(model_decoder.parameters(), lr=cfg.TRAIN.LR)

    elif cfg.TRAIN.POLICY == 'sgd':
        model_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LR,
                                       momentum=cfg.TRAIN.MOMENTUM)
        model_decoder_solver = torch.optim.SGD(model_decoder.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM)

    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    model_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_solver,
                                                              milestones=cfg.TRAIN.MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)
    model_decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_decoder_solver,
                                                                      milestones=cfg.TRAIN.MILESTONES,
                                                                      gamma=cfg.TRAIN.GAMMA)

    # model_lr_scheduler = lr_scheduler.CosineAnnealingLR(model_solver, cfg.TRAIN.NUM_EPOCHS, cfg.TRAIN.MIN_LR)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        model_decoder = torch.nn.DataParallel(model_decoder).cuda()

    # Set up loss functions
    loss_func = utils.helpers.get_loss_function(cfg, cfg.NETWORK.LOSS_FUNC)

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        model.load_state_dict(checkpoint['model_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())

    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):

        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model_losses = AverageMeter()

        # switch models to training mode
        model.train()
        model_decoder.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes).long()

            # Train the model, decoder, refiner, and merger
            image_features = model(rendering_images)
            raw_features, generated_volumes = model_decoder(image_features)

            model_loss = loss_func(generated_volumes,
                                   torch.nn.functional.one_hot(ground_truth_volumes.long()).float().permute(0, -1, 1, 2,
                                                                                                            3)) * 10

            # Gradient decent
            model.zero_grad()
            model_decoder.zero_grad()

            model_loss.backward()

            model_solver.step()
            model_decoder_solver.step()

            # Append loss to average metrics
            model_losses.update(model_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('modelDecoder/BatchLoss', model_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f' %
                (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                 model_loss.item()))

        # del generated_volumes, rendering_images, ground_truth_volumes

        # Adjust learning rate
        model_lr_scheduler.step()
        model_decoder_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('model/EpochLoss', model_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, model_losses.avg))

        # Update Rendering Views
        # if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
        #     n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
        #     train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
        #     logging.info('Epoch [%d/%d] Update #RenderingViews to %d' %
        #                  (epoch_idx + 2, cfg.TRAIN.NUM_EPOCHS, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, model=model, model_decoder=model_decoder,
                       best_total_iou=best_iou)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)

            with open(os.path.join(cfg.DIR.OUT_PATH, 'train_test_config.json'), 'w') as fp:
                json.dump(cfg, fp, indent=4)
                fp.close()

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict()
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    train_writer.close()
    val_writer.close()
