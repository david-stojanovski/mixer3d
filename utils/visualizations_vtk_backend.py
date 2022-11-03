import pyvista as pv
import numpy as np
import vtk
from vtk.util import numpy_support

def move_to_origin(surface, center=None):
    if not center:
        centerFilter = vtk.vtkCenterOfMass()
        centerFilter.SetInputData(surface)
        centerFilter.SetUseScalarsAsWeights(False)
        centerFilter.Update()
        center = centerFilter.GetCenter()

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetInputData(surface)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    centeredSurface = transformFilter.GetOutput()

    return centeredSurface


def rotate_mesh(polydata, angle, axis='z'):
    if axis == 'z':
        axis = [0, 0, 1]
    elif axis == 'y':
        axis = [0, 1, 0]
    elif axis == 'x':
        axis = [1, 0, 0]

    transform = vtk.vtkTransform()
    transform.RotateWXYZ(angle, axis[0], axis[1], axis[2])
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polydata)
    transformFilter.Update()
    return transformFilter.GetOutput()


def center_of_mass(surface):
    """ Get center of mass of Polydata """
    centerFilter = vtk.vtkCenterOfMass()
    centerFilter.SetInputData(surface)
    centerFilter.SetUseScalarsAsWeights(False)
    centerFilter.Update()
    center = centerFilter.GetCenter()
    return center


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def point_threshold(polydata, arrayname, start=0, end=1, alloff=0):
    """ Clip polydata according to given threshold in scalar array"""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    if vtk.vtkVersion.GetVTKMajorVersion() > 8:
        threshold.SetLowerThreshold(start)
        threshold.SetUpperThreshold(end)
    else:
        threshold.ThresholdBetween(start, end)
    if alloff:
        threshold.AllScalarsOff()
    threshold.Update()
    # surfer = surfer_filter(threshold.GetOutput())
    return threshold.GetOutput()


def cell_threshold(polydata, arrayname, start=0, end=1):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arrayname)
    if vtk.vtkVersion.GetVTKMajorVersion() > 8:
        threshold.SetLowerThreshold(start)
        threshold.SetUpperThreshold(end)
    else:
        threshold.ThresholdBetween(start, end)
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()


def rotate_and_align_mesh_cell(cfg, mesh):
    com = center_of_mass(mesh)
    mesh = move_to_origin(mesh, com)

    com_lv = center_of_mass(cell_threshold(mesh, cfg.LABELS.LABEL_NAME, 1, 1))
    com_la = center_of_mass(cell_threshold(mesh, cfg.LABELS.LABEL_NAME, 3, 3))
    vector = np.array(com_la) - np.array(com_lv)

    rotation_matrix = rotation_matrix_from_vectors(vector, [0, 0, 1])
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_matrix
    rotation[3, 3] = 1

    mesh = pv.wrap(mesh).transform(rotation, transform_all_input_vectors=True)

    # align with respect to rv/lv direction
    com_lv = center_of_mass(cell_threshold(mesh, cfg.LABELS.LABEL_NAME, 1, 1))
    com_rv = center_of_mass(cell_threshold(mesh, cfg.LABELS.LABEL_NAME, 2, 2))
    vector = np.array(com_rv) - np.array(com_lv)
    vector[2] = 0
    rotation_angle = angle_between(vector, [-1, 0, 0])
    rotated_mesh = rotate_mesh(mesh, rotation_angle, 'z')
    rotated_mesh = pv.wrap(rotated_mesh).scale([-1, -1, 1], inplace=True)
    return rotated_mesh, rotation, rotation_angle, com


def rotate_and_align_mesh(cfg, mesh):
    com = center_of_mass(mesh)
    mesh = move_to_origin(mesh, com)

    com_lv = center_of_mass(point_threshold(mesh, cfg.LABELS.METADATA.LABEL_NAME, 1, 1))
    com_la = center_of_mass(point_threshold(mesh, cfg.LABELS.METADATA.LABEL_NAME, 3, 3))
    vector = np.array(com_la) - np.array(com_lv)

    rotation_matrix = rotation_matrix_from_vectors(vector, [0, 0, 1])
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_matrix
    rotation[3, 3] = 1

    mesh = pv.wrap(mesh).transform(rotation, transform_all_input_vectors=True)

    # align with respect to rv/lv direction
    com_lv = center_of_mass(point_threshold(mesh, 'elemTag', 1, 1))
    com_rv = center_of_mass(point_threshold(mesh, 'elemTag', 2, 2))
    vector = np.array(com_rv) - np.array(com_lv)
    vector[2] = 0
    rotation_angle = angle_between(vector, [-1, 0, 0])
    rotated_mesh = rotate_mesh(mesh, rotation_angle, 'z')
    rotated_mesh = pv.wrap(rotated_mesh).scale([-1, -1, 1], inplace=True)
    return rotated_mesh, rotation, rotation_angle, com


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_rad = np.arccos(np.dot(v1_u, v2_u))
    angle = np.rad2deg(angle_rad)
    return angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def read_predicted_np_volume(npy_path):
    npy_vol = np.load(npy_path)
    out_volume = np.squeeze(np.argmax(npy_vol, axis=1))  # gets integer classes from logits
    return out_volume


def add_label_id_to_surface(surface, label_array_name, label_value):
    n_points = surface.GetNumberOfPoints()
    label = vtk.vtkIntArray()
    label.SetNumberOfComponents(0)
    label.SetName(label_array_name)
    for j in range(0, n_points):
        label.InsertNextValue(label_value + 1)
    surface.GetPointData().AddArray(label)
    return surface


def vtk_discrete_marching_cubes(segmentation, surfaceValue=1):
    """Return a surface model from a segmentation.
    The surface is generated by marching cubes algorithm.
    The surface is generated from the voxels that have a value of ``surfaceValue``.
    Parameters
    ----------
    segmentation : vtk.vtkImageData
        Segmentation volume.
    surfaceValue : int
        Value of the voxels to be included in the surface.
    Returns
    -------
    surface : vtk.vtkPolyData
        Surface model.
    """
    # Create a model from the segmentation
    surface = vtk.vtkDiscreteMarchingCubes()
    surface.SetInputData(segmentation)
    surface.SetValue(0, surfaceValue)
    surface.Update()
    return surface.GetOutput()


def generate_mesh_from_volume(cfg, volume, save_path=None):
    for single_class in np.unique(volume):
        if single_class != 0:
            surface_mesh = vtk_discrete_marching_cubes(pv.wrap(volume), surfaceValue=single_class)
            surface_mesh = add_label_id_to_surface(surface_mesh, cfg.LABELS.METADATA.LABEL_NAME, single_class - 1)
            smoothed = refine_polydata(surface_mesh, iterations=50, set_passband=True)
            if single_class == 1:
                out_polydata = smoothed
            else:
                out_polydata = append_polydata(out_polydata, smoothed)
    if save_path is not None:
        pv.wrap(out_polydata).save(save_path)
    return out_polydata


def append_polydata(polydata1, polydata2):
    """
    Append two polydata
    To obtain cell/point fields in the output,
    both inputs need to have the same fields
    """
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(polydata1)
    appendFilter.AddInputData(polydata2)
    appendFilter.Update()
    surface = appendFilter.GetOutput()
    return surface


def refine_polydata(polydata, iterations=30, set_passband=False):
    """
    - Adjusts point positions using a windowed sinc function interpolation kernel.
    - The effect is to "relax" the mesh, making the cells better shaped and the vertices more evenly distributed.
    """
    refiner = vtk.vtkWindowedSincPolyDataFilter()
    refiner.SetInputData(polydata)
    refiner.SetNumberOfIterations(iterations)
    refiner.NonManifoldSmoothingOn()
    if set_passband:
        refiner.SetPassBand(0.05)
    refiner.NormalizeCoordinatesOff()
    refiner.GenerateErrorScalarsOff()
    refiner.Update()
    return refiner.GetOutput()


def convert_to_mesh(cfg, predicted_volume, gt_volume=None, save_path=None):
    predicted_volume1 = np.squeeze(np.argmax(predicted_volume, axis=1))
    predicted_mesh = generate_mesh_from_volume(cfg, predicted_volume1, save_path=None)
    transformed_prediction, rotation, rotation_angle, com = rotate_and_align_mesh(cfg, predicted_mesh)
    transformed_prediction = pv.wrap(transformed_prediction).scale(
        [cfg.LABELS.METADATA.VOXEL_SCALING, cfg.LABELS.METADATA.VOXEL_SCALING, cfg.LABELS.METADATA.VOXEL_SCALING],
        inplace=True)

    if gt_volume is not None:
        gt_mesh = generate_mesh_from_volume(cfg, gt_volume.astype(int), save_path=None)
        gt_mesh = move_to_origin(gt_mesh, com)
        gt_mesh = pv.wrap(gt_mesh).transform(rotation, transform_all_input_vectors=True)
        gt_mesh = rotate_mesh(gt_mesh, rotation_angle, 'z')
        gt_mesh = pv.wrap(gt_mesh).scale([-1, -1, 1], inplace=True)
        gt_mesh = pv.wrap(gt_mesh).scale([cfg.LABELS.METADATA.VOXEL_SCALING, cfg.LABELS.METADATA.VOXEL_SCALING, cfg.LABELS.METADATA.VOXEL_SCALING],
                                         inplace=True)
        return transformed_prediction, gt_mesh
    else:
        return transformed_prediction
