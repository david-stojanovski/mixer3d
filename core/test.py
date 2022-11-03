# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import logging
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.visualizations
from utils.average_meter import AverageMeter
from utils.metrics import MeanIoU
from models.archs import ConvMixer, Decoder
from core.inference import inference_net


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             model=None,
             model_decoder=None,
             best_total_iou=None):
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        test_transforms = utils.data_transforms.Compose([
            # utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)

    # Set up networks
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

    # Set up loss functions
    loss_func_model = utils.helpers.get_loss_function(cfg, cfg.NETWORK.LOSS_FUNC)
    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    model_losses = AverageMeter()

    # Switch models to evaluation mode
    model.eval()
    model_decoder.eval()

    all_iou = []
    myo_diff_total = []
    myo_vol_total = []
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]


        with torch.no_grad():
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            image_features = model(rendering_images)
            raw_feature, generated_volume = model_decoder(image_features)

            model_loss = loss_func_model(generated_volume,
                                         torch.nn.functional.one_hot(ground_truth_volume.long()).float().permute(0, -1,
                                                                                                                 1, 2,
                                                                                                                 3)) * 10

            model_losses.update(model_loss.item())

            sample_iou = []

            class_mean_sample_iou, class_all_sample_iou = MeanIoU(torch.nn.functional.softmax(generated_volume, dim=1),
                                                                  ground_truth_volume.long())
            sample_iou.append(class_mean_sample_iou)
            all_iou.append(class_all_sample_iou)

            myocardium_volumes, myocardium_differences = utils.metrics.calc_biomarkers(cfg, generated_volume,
                                                                                       ground_truth_volume)
            myo_diff_total += [myocardium_differences]
            myo_vol_total += [myocardium_volumes]

            logging.info('Sample = %s EDLoss = %.4f IoU = %s' %
                         (sample_name, model_loss.item(), np.mean(class_all_sample_iou)))

    # IoU per taxonomy
    if taxonomy_id not in test_iou:
        test_iou[taxonomy_id] = {'n_samples': 0, 'all_iou': [], 'iou': []}
    test_iou[taxonomy_id]['n_samples'] += 1
    test_iou[taxonomy_id]['iou'].append(sample_iou)
    test_iou[taxonomy_id]['all_iou'].append(all_iou)

    all_iou_arr = np.squeeze(np.array(all_iou))
    single_mean_iou = np.mean(all_iou_arr)
    per_case_mean_iou = np.mean(all_iou_arr, axis=1)
    # Print sample loss and IoU
    logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f IoU = %s' %
                 (sample_idx + 1, n_samples, taxonomy_id, sample_name, model_loss.item(),
                  ['%.4f' % si for si in sample_iou]))

    if cfg.DIR.IOU_SAVE_PATH:
        df = pd.DataFrame(np.hstack((all_iou_arr, np.atleast_2d(per_case_mean_iou).T)),
                          columns=[*range(0, cfg.CONST.NUM_CLASSES), 'mean_iou'])
        writer = pd.ExcelWriter(cfg.DIR.IOU_SAVE_PATH, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.close()

    # Print header
    print('============================ TEST RESULTS ============================')
    # print('Taxonomy', end='\t')
    # print('#Sample', end='\t')
    # for th in cfg.TEST.VOXEL_THRESH:
    #     print('t=%.2f' % th, end='\t')
    # print()
    # Print body
    # for taxonomy_id in test_iou:
    #     print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
    #     print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
    #     for ti in test_iou[taxonomy_id]['iou']:
    #         print('%.4f' % ti, end='\t')
    #     print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    # for mi in single_mean_iou:
    print('%.4f' % single_mean_iou, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    # max_iou = np.max(mean_iou)
    max_iou = single_mean_iou
    if max_iou > best_total_iou:
        print('Saving generated volumes...')
        for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(
                test_data_loader):

            print('Saving volume for %s ...' % sample_name[0])
            with torch.no_grad():
                # Get data from data loader
                rendering_images = utils.helpers.var_or_cuda(rendering_images)
                ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

                # Test the model, decoder, refiner and merger
                image_features = model(rendering_images)
                raw_feature, generated_volume = model_decoder(image_features)

                generated_volume = generated_volume.detach().cpu().numpy()
                ground_truth_volume = torch.squeeze(ground_truth_volume).cpu().numpy()

                if cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'render':
                    if test_writer and sample_idx < cfg.CONST.TEST_SAVE_NUMBER:
                        # Volume Visualization
                        rendering_views = utils.visualizations.get_volume_views(generated_volume.cpu().numpy())
                        test_writer.add_image('Model%02d/Reconstructed' % sample_name[0], rendering_views, epoch_idx)
                        rendering_views = utils.visualizations.get_volume_views(ground_truth_volume.cpu().numpy())
                        test_writer.add_image('Model%02d/GroundTruth' % sample_name[0], rendering_views, epoch_idx)
                elif cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'volume':
                    # if test_writer and sample_idx < cfg.CONST.TEST_SAVE_NUMBER:
                    utils.visualizations.save_test_volumes_as_np(cfg, generated_volume, sample_name[0])
                else:
                    raise Exception(
                        '[FATAL] %s Invalid input for save format %s. voxels' % (dt.now(), cfg.TEST.VOL_OR_RENDER_SAVE))

                if cfg.CONST.SAVE_MONTAGE and not cfg.CONST.MULTI_LABEL:
                    utils.visualizations.show_model_diff(cfg, generated_volume, ground_truth_volume, sample_name[0])

                elif cfg.CONST.SAVE_MONTAGE and cfg.CONST.MULTI_LABEL:

                    utils.visualizations.generate_montage(cfg, generated_volume, ground_truth_volume,
                                                          sample_name[0])

                if cfg.CONST.SAVE_VTK_MESHES or cfg.CONST.SAVE_MESH_PLOTS:
                    myo_vols_zero_check = np.where(np.all(np.isclose(np.squeeze(myo_vol_total)[:3], 0),
                                                          axis=1))  # [:3] to avoid check of right atrium
                    if len(myo_vols_zero_check[0]) == 0 and epoch_idx >= cfg.CONST.MIN_SAVE_VTK_MESHES_EPOCH:
                        utils.visualizations.plot_clipped_mesh(cfg, generated_volume, ground_truth_volume,
                                                               sample_name[0])

        if cfg.TRAIN.TRAIN_TIME_INFERENCE and epoch_idx >= cfg.CONST.MIN_SAVE_VTK_MESHES_EPOCH:
            mean_volume_prediction = inference_net(cfg, epoch_idx + 1, model=model, model_decoder=model_decoder)
            if test_writer is not None:
                if cfg.TRAIN.TRAIN_TIME_INFERENCE:
                    test_writer.add_scalar('lv_myocardium_mean', mean_volume_prediction[0], epoch_idx)
                    test_writer.add_scalar('rv_myocardium_mean', mean_volume_prediction[1], epoch_idx)
                    test_writer.add_scalar('la_myocardium_mean', mean_volume_prediction[2], epoch_idx)
                    test_writer.add_scalar('ra_myocardium_mean', mean_volume_prediction[3], epoch_idx)

    if test_writer is not None:
        test_writer.add_scalar('model/EpochLoss', model_losses.avg, epoch_idx)
        test_writer.add_scalar('IoU', max_iou, epoch_idx)

        myo_diff_total = np.mean(np.squeeze(myo_diff_total), axis=1)
        test_writer.add_scalar('lv_myocardium_differences', myo_diff_total[0], epoch_idx)
        test_writer.add_scalar('rv_myocardium_differences', myo_diff_total[1], epoch_idx)
        test_writer.add_scalar('la_myocardium_differences', myo_diff_total[2], epoch_idx)
        test_writer.add_scalar('ra_myocardium_differences', myo_diff_total[3], epoch_idx)

    return max_iou
