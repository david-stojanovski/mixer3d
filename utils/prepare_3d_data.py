import glob
import os
import shutil

from natsort import natsorted

img_root_path = r'/home/ds17/Documents/phd/p2vse_a/datasets/heart_seg/heart_render_binary_4/heart/'
volume_root_path = r'/home/ds17/Documents/phd/p2vse_a/datasets/heart_seg/voxel_vtk_128/heart/'
img_save_dir = r'/home/ds17/Documents/phd/UNeXt-pytorch/inputs/heart/images'
volume_save_dir = r'/home/ds17/Documents/phd/UNeXt-pytorch/inputs/heart/masks'

img_paths = natsorted(glob.glob(os.path.join(img_root_path, '*', '*.png')))
volume_paths = natsorted(glob.glob(os.path.join(volume_root_path, '*', 'model.npy')))

for img_path in img_paths:
    case_name = img_path.split(os.sep)[-2]
    save_path = os.path.join(img_save_dir, case_name + '.png')
    shutil.copy2(img_path, save_path)

for volume_path in volume_paths:
    case_name = volume_path.split(os.sep)[-2]
    save_path = os.path.join(volume_save_dir, case_name + '.npy')
    shutil.copy2(volume_path, save_path)
