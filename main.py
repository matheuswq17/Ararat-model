import numpy as np
import SimpleITK as sitk
import os

from radiomics.featureextractor import RadiomicsFeatureExtractor

def create_segmentation(dicom_folder, center_mm, radius_mm=5, result_file="mask_temp.nii.gz"):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    img_np = sitk.GetArrayFromImage(image)
    
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    voxel_center = [(center_mm[i] - origin[i]) / spacing[i] for i in range(3)]
    voxel_center = np.round(voxel_center).astype(int)

    mask = np.zeros_like(img_np, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    distance = (
        (xx - voxel_center[0])**2 +
        (yy - voxel_center[1])**2 +
        ((zz - voxel_center[2])**2)**0.5
    )

    mask[distance <= radius_mm / spacing[0]] = 1

    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetSpacing(spacing)
    mask_itk.SetOrigin(origin)
    mask_itk.SetDirection(direction)
    
    sitk.WriteImage(mask_itk, result_file)

    return image

def get_features(image_path, roi_path):
    extractor = RadiomicsFeatureExtractor()

    extractor.settings['geometryTolerance'] = 1e-5

    features = extractor.execute(image_path, roi_path)

    for k, v in features.items():
        print(f"{k}: {v}")

x = -8.58088
y = 26.3826
z = 26.3826

coords = [x, y, z]
folder_path = 'data'

image = create_segmentation(folder_path, coords)
mask = sitk.ReadImage('mask_temp.nii.gz')


get_features(image, mask)