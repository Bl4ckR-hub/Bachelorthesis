import os
from torch.utils.data import Dataset
import torch
import h5py
import odl
import torch.nn.functional as F
import numpy as np

class LoDoPaB_Dataset(Dataset):
    def __init__(self, sino_dir, gt_images_dir, transform=None, target_transform=None, suffix=None, amount_images=None):
        self.gt_image_names = sorted([x for x in os.listdir(gt_images_dir) if 'ground_truth' in x])
        self.sino_names = sorted([x for x in os.listdir(sino_dir) if 'observation' in x])

        if suffix:
            self.gt_image_names = sorted([x for x in self.gt_image_names if suffix in x])
            self.sino_names = sorted([x for x in self.sino_names if suffix in x])

        self.gt_image_files = [os.path.join(gt_images_dir, x) for x in self.gt_image_names]
        self.sino_files = [os.path.join(sino_dir, x) for x in self.sino_names]


        # Assume each file contains 128 slices except maybe the last
        self.index_map = []
        for file_idx, (gt_path, sino_path) in enumerate(zip(self.gt_image_files, self.sino_files)):
            if type(amount_images) is int and amount_images >= 0 and (file_idx)*128 >= amount_images:
                break
            with h5py.File(gt_path, 'r') as gt_file:
                n_items = len(gt_file['data'])
                for i in range(n_items):
                    if type(amount_images) is int and amount_images >= 0 and (file_idx) * 128 + i >= amount_images:
                        break
                    self.index_map.append((file_idx, i))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, data_idx = self.index_map[idx]

        with h5py.File(self.sino_files[file_idx], 'r') as sino_file:
            sino = torch.from_numpy(sino_file['data'][data_idx])[None, :, :]

        with h5py.File(self.gt_image_files[file_idx], 'r') as gt_file:
            gt_image = torch.from_numpy(gt_file['data'][data_idx])[None, :, :]

        if self.transform:
            sino = self.transform(sino)
        if self.target_transform:
            gt_image = self.target_transform(gt_image)

        return sino, gt_image


##############################################################
def crop_zoom_top_left(image: torch.Tensor, x: int, y: int, width: int, height: int):
    """
    Crops a zoomed-in region from a grayscale image tensor using top-left coordinates.
    Args:
        image (torch.Tensor): 2D tensor (H, W) or 3D tensor (C, H, W)
        x (int): x-coordinate (column) of the top-left corner
        y (int): y-coordinate (row) of the top-left corner
        width (int): width of the crop
        height (int): height of the crop
    Returns:
        torch.Tensor: Cropped image tensor
    """
    if image.dim() == 2:
        return image[y:y+height, x:x+width]
    elif image.dim() == 3:
        return image[:, y:y+height, x:x+width]
    else:
        raise ValueError("Input image must be 2D or 3D tensor.")
##############################################################

def min_max_normalize(x, eps=1e-8):
    # x: (B, 1, H, W) or (B, H, W)
    x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_norm = (x - x_min) / (x_max - x_min + eps)
    return x_norm


def gt_to_coeffs(Y, max=81.35858):
    return Y * max

def X_to_minuspostlog(X, max=81.35858):
    return (X * max)

def minuspostlog_to_proj(X, N_0=4096):
    return torch.exp(-X)*N_0


class RadonTransform():
    def __init__(self, device=torch.device('cpu')):
        space = odl.uniform_discr(min_pt=[-0.13, -0.13], max_pt=[+0.13, +0.13], shape=(1000, 1000), dtype='float32')
        angle_partition = odl.uniform_partition(0, np.pi, 1000)  # 0 to pi radians
        detector_length = (0.26**2 + 0.26**2)**(1/2)
        detector_partition = odl.uniform_partition(-detector_length/2, +detector_length/2, 513)  # detector length in meters

        geometry = odl.tomo.Parallel2dGeometry(
            apart=angle_partition,
            dpart=detector_partition
        )
        self.ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu' if device == torch.device('cpu') else 'astra_cuda')

    def radon(self, Y_coeffs):
        Y_inter = F.interpolate(Y_coeffs, size=(1000, 1000)).detach().cpu().numpy()
        X_radon = torch.Tensor(np.stack([self.ray_trafo(Y_inter[i][0]) for i in range(Y_inter.shape[0])])).unsqueeze(1).to()
        return X_radon
    
def min_max_norm(X, eps=1e-9):
    v_max = torch.max(torch.max(X, dim=2).values, dim=2).values.unsqueeze(0).unsqueeze(0)
    v_min = torch.min(torch.min(X, dim=2).values, dim=2).values.unsqueeze(0).unsqueeze(0)
    return (X - v_min)/(v_max - v_min + eps)

def norm_to_classic(X, v_min, v_max, eps=1e-9):
    return X*(v_max - v_min + eps) + v_min