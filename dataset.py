from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from torchvision import tv_tensors

def read_xray(dir):
    xray = cv2.imread(dir , cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32)/255.0
    xray = xray.reshape(1, *xray.shape)
    return xray

def read_mask(dir):
    mask = cv2.imread(dir , cv2.IMREAD_GRAYSCALE)
    mask = (mask>0).astype(np.float32)
    mask = mask.reshape(1, *mask.shape)
    return mask

class KneeDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray_path = self.df['xrays'].iloc[index]
        mask_path = self.df['masks'].iloc[index]

        xray = read_xray(xray_path)  # X Value

        if isinstance(mask_path, str):
            mask = read_mask(mask_path)  # Y Value
        else:
            mask = np.zeros_like(xray)  # test images have no mask

        xray = tv_tensors.Image(torch.from_numpy(xray))
        mask = tv_tensors.Mask(torch.from_numpy(mask))

        if self.transform is not None:
            xray, mask = self.transform(xray, mask)  # SAME random decision for both

        res = {
            "xray": xray,
            "mask": mask,
            "path": xray_path
        }

        return res






