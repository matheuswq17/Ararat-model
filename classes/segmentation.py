import SimpleITK as sitk
import numpy as np

class Segmentation():
    
    @staticmethod
    def create_image_mask(image_sitk: sitk.Image, center_mm: list, radius_mm: int = 5) -> sitk.Image:
        """
        cria uma máscara binaria esferica 3D em memoria a partir de uma imagem de referencia
        """
        voxel_center_continuous = image_sitk.TransformPhysicalPointToContinuousIndex(center_mm)
        zz, yy, xx = np.mgrid[:image_sitk.GetDepth(), :image_sitk.GetHeight(), :image_sitk.GetWidth()]
        spacing = np.array(image_sitk.GetSpacing())
        
        distance_sq = (
            (spacing[0] * (xx - voxel_center_continuous[0]))**2 +
            (spacing[1] * (yy - voxel_center_continuous[1]))**2 +
            (spacing[2] * (zz - voxel_center_continuous[2]))**2
        )
        
        mask_np = (distance_sq <= radius_mm**2).astype(np.uint8)
        mask_itk = sitk.GetImageFromArray(mask_np)
        mask_itk.CopyInformation(image_sitk)
        
        return mask_itk