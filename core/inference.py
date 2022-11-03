# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
from datetime import datetime as dt

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.visualizations
from models.archs import ConvMixer, Decoder
from utils.metrics import calc_biomarkers


def inference_net(cfg,
                  epoch_idx=-1,
                  inference_data_loader=None,
                  test_writer=None,
                  model=None,
                  model_decoder=None,):
    torch.backends.cudnn.benchmark = True

    if inference_data_loader is None:
        inference_transforms = utils.data_transforms.Compose([
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.INFERENCE_DATASET](cfg)
        inference_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(cfg.CONST.N_VIEWS_RENDERING, inference_transforms),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)

    if model is None or model_decoder is None:
        model = ConvMixer(cfg)
        model_decoder = Decoder(cfg)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            model_decoder = torch.nn.DataParallel(model_decoder).cuda()

        logging.info('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model_decoder.eval()

    myo_vol_total = []
    for sample_idx, (taxonomy_id, sample_name, rendering_images) in enumerate(inference_data_loader):
        sample_name = sample_name[0]



        print('Processing %s ...' % sample_name)
        with torch.no_grad():
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            image_features = model(rendering_images)
            raw_feature, generated_volume = model_decoder(image_features)


            myocardium_volumes = calc_biomarkers(cfg, generated_volume)
            myo_vol_total += [myocardium_volumes]
            generated_volume = generated_volume.detach().cpu().numpy()
            # if cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'render':
            #     if test_writer and sample_idx < cfg.CONST.TEST_SAVE_NUMBER:
            #         rendering_views = utils.visualizations.get_volume_views(generated_volume.cpu().numpy())
            #         test_writer.add_image('Model%02d/Reconstructed' % sample_name, rendering_views, epoch_idx)
            # elif cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'volume':
            #     utils.visualizations.save_test_volumes_as_np(cfg, generated_volume, sample_name, inference_mode=True)
            # else:
            #     raise Exception(
            #         '[FATAL] %s Invalid input for save format %s. voxels' % (dt.now(), cfg.TEST.VOL_OR_RENDER_SAVE))

            if cfg.CONST.SAVE_VTK_MESHES or cfg.CONST.SAVE_MESH_PLOTS:
                myo_vols_zero_check = np.where(np.all(np.isclose(np.squeeze(myocardium_volumes)[:3], 0),
                                                      axis=0))  # [:3] to avoid check of right atrium
                if len(myo_vols_zero_check[0]) == 0 and epoch_idx >= cfg.CONST.MIN_SAVE_VTK_MESHES_EPOCH:
                    utils.visualizations.plot_clipped_mesh(cfg, generated_volume, img_name=sample_name, inference_mode=True)
                else:
                    logging.info('[INFO] %s Skipping mesh plot for %s due to zero volumes.' % (dt.now(), sample_name))

    return np.mean(np.squeeze(myo_vol_total), axis=0)
