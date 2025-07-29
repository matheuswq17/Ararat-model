import SimpleITK as sitk
import numpy as np

class Segmentation():
    
    @staticmethod
    def process(dicom_path, center_mm, radius_mm=5, result_path="mask_temp.nii.gz"):
        """
            Args:
                dicom_path: Dicom images folder path
                center_mm: ROIs spheric center
                radius_mm: ROIs spheric radius
                result_path: Final ROI file path
        """
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_path)
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, series_IDs[0])
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

        # TODO: This creates a cilinder, not a sphere
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
        
        sitk.WriteImage(mask_itk, result_path)

        return image