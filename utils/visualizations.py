from matplotlib.colors import ListedColormap
from skimage.util import montage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import matplotlib
import pyvista as pv
import utils.visualizations_vtk_backend as vis_vtk

matplotlib.use('Qt5Agg')


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def save_test_volumes_as_np(cfg, volume, sample_name, inference_mode=False):
    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'volumes')
    if inference_mode:
        test_case_path = os.path.join(img_dir, 'inference')
    else:
        test_case_path = os.path.join(img_dir, 'test')
    save_path = os.path.join(test_case_path, str(sample_name) + os.sep)

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(test_case_path):
        os.mkdir(test_case_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    np.save(save_path + 'best_epoch', volume)


def generate_montage(cfg, reconstructed_model, gt_model, img_name=None):
    reconstructed_model = np.argmax(np.squeeze(reconstructed_model), axis=0, keepdims=False)
    in_model_montage = montage(reconstructed_model, fill=0, padding_width=0, grid_shape=(8, 16))

    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'volumes')
    test_case_path = os.path.join(img_dir, 'test')
    save_path = os.path.join(test_case_path, str(img_name) + os.sep)
    matplotlib.image.imsave(save_path + 'best_epoch_montage_labels.png', in_model_montage)

    show_model_diff(cfg, reconstructed_model, np.squeeze(gt_model), img_name)


def show_model_diff(cfg, reconstructed_model, gt_model, img_name=None):
    reconstructed_model = (reconstructed_model > 0).astype(np.int_)
    gt_model = (gt_model > 0).astype(np.int_)

    diff = gt_model + 1 - reconstructed_model * 2
    diff_mon = montage(diff, fill=0, padding_width=0, grid_shape=(8, 16))

    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'volumes')
    test_case_path = os.path.join(img_dir, 'test')
    save_path = os.path.join(test_case_path, str(img_name) + os.sep)

    cmap_custom = ListedColormap(['r', 'g', 'k', 'b'])
    matplotlib.image.imsave(save_path + 'best_epoch_montage.png', diff_mon, cmap=cmap_custom)


def plot_clipped_mesh(cfg, predicted_volume, ground_truth_volume=None, img_name=None, inference_mode=False):
    if ground_truth_volume is not None:
        predicted_mesh, ground_truth_mesh = vis_vtk.convert_to_mesh(cfg, predicted_volume, ground_truth_volume)
    else:
        predicted_mesh = vis_vtk.convert_to_mesh(cfg, predicted_volume)

    clipped_heart_2 = predicted_mesh.clip(normal='y', invert=True)

    plotter = pv.Plotter(shape=(1, 2), off_screen=True)
    plotter.set_background('white')

    plotter.subplot(0, 0)
    plotter.camera.up = (1.0, 1.0, 1.0)
    plotter.camera.position = [0, 100, 0]
    plotter.camera.zoom(0.2)
    plotter.camera.roll = 180
    plotter.add_mesh(predicted_mesh,  # Extracted slice
                     scalars=cfg.LABELS.METADATA.LABEL_NAME,  # Color by the scalars
                     smooth_shading=True,
                     show_scalar_bar=False)
    plotter.enable_parallel_projection()

    plotter.subplot(0, 1)
    plotter.set_focus([0, 0, 0])
    plotter.camera.up = (1.0, -1.0, 1.0)
    plotter.camera.position = [0, 100, 0]
    plotter.camera.zoom(0.2)
    plotter.camera.roll = 180
    plotter.add_mesh(clipped_heart_2,  # Extracted slice
                     scalars=cfg.LABELS.METADATA.LABEL_NAME,  # Color by the scalars
                     smooth_shading=True,
                     show_scalar_bar=False)
    plotter.enable_parallel_projection()

    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'volumes')
    if inference_mode:
        test_case_path = os.path.join(img_dir, 'inference')
    else:
        test_case_path = os.path.join(img_dir, 'test')
        save_path = os.path.join(test_case_path, str(img_name) + os.sep)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    if cfg.CONST.SAVE_VTK_MESHES and not inference_mode:
        predicted_mesh.save(save_path + 'prediction_mesh.vtk')
    elif cfg.CONST.SAVE_VTK_MESHES and inference_mode:
        save_path = os.path.join(test_case_path, str(img_name) + os.sep)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        predicted_mesh.save(save_path + 'prediction_mesh.vtk')
    if ground_truth_volume is not None:
        ground_truth_mesh.save(save_path + 'groundtruth_mesh.vtk')

    if cfg.CONST.SAVE_MESH_PLOTS and not inference_mode:
        plotter.screenshot(save_path + 'clipped_prediction.png', return_img=True)
    elif cfg.CONST.SAVE_MESH_PLOTS and inference_mode:
        if not os.path.isdir(test_case_path):
            os.makedirs(test_case_path)
        plotter.screenshot(os.path.join(test_case_path,  str(img_name) + '_clipped_prediction.png'), return_img=True)

    plotter.close()
    return
