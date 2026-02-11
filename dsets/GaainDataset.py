import os
import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import pandas as pd
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchsummary import summary
from natsort import natsorted
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk
import tqdm
from importlib import reload
from functools import lru_cache
from utils.tools import CLASSES, COLORDICT
from utils.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



class GaainDataset(BaseDataset):
    CLASSES = CLASSES
    COLORDICT = COLORDICT
    
    def __init__(self, patient_dir, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = sorted(os.listdir(patient_dir))
        self.images_fps = [os.path.join(patient_dir, image_id, images_dir) for image_id in self.ids]
        self.masks_fps = [os.path.join(patient_dir, image_id, masks_dir) for image_id in self.ids]

        flipped = {value: key for key, value in self.CLASSES.items()}
        self.index_classes = sorted([flipped[el] for el in classes])
        
        img_notfound = False
        msk_notfound = False
        for idx, (img, msk) in enumerate(zip(self.images_fps, self.masks_fps)):
            if not os.path.exists(img):
                if os.path.exists(img+'.gz'):
                    self.images_fps[idx] = img+'.gz'
                else:
                    img_notfound = True
                    log.warn(f'{self.ids[idx]}: image not found')
            if not os.path.exists(msk):
                if os.path.exists(msk+'.gz'):
                    self.masks_fps[idx] = msk+'.gz'
                else:
                    msk_notfound = True
                    log.warn(f'{self.ids[idx]}: mask not found')
            if img_notfound or msk_notfound:
                log.warn(f'{self.ids[idx]}: Removed from patient list')
                del self.ids[idx]
                del self.images_fps[idx]
                del self.masks_fps[idx]
                img_notfound = False
                msk_notfound = False
        
        self.class_values = [key for key, val in self.CLASSES.items() if val.lower() in [el.lower() for el in classes]] if classes else None

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __len__(self):
        return len(self.ids)

    def get_pid(self, idx):
        return self.ids[idx]
    
    def __getitem__(self, idx):
        image_path = self.images_fps[idx]
        mask_path = self.masks_fps[idx]
        itk_image = sitk.ReadImage(image_path)
        itk_mask = sitk.ReadImage(mask_path)
        numpy_image = sitk.GetArrayFromImage(itk_image).astype('float32')
        numpy_mask = sitk.GetArrayFromImage(itk_mask).astype('float32')
        torch_image = torch.from_numpy(numpy_image)
        torch_mask = torch.from_numpy(numpy_mask)
        
        if self.class_values:
            boolean_masks = [(torch_mask == v) for v in self.class_values]
            sum_mask = torch.stack(boolean_masks).sum(dim=0).clamp(max=1).float()
            torch_mask *= sum_mask
            
        one_hot_masks, _ = self.one_hot_encode(torch_mask)

        if self.preprocessing:
            pass

        if self.augmentation:
            subject_dict = {
                'volume': tio.ScalarImage(tensor=torch_image.unsqueeze(0)), 
                'mask': tio.LabelMap(tensor=torch_mask.unsqueeze(0)),
            }
            
            for i in range(one_hot_masks.shape[0]):
                key = f'one_hot_label_{i}'
                value = one_hot_masks[i].unsqueeze(0)
                subject_dict[key] = tio.LabelMap(tensor=value)
            
            subject = tio.Subject(**subject_dict)
            transformed_subject = self.augmentation(subject)
            torch_image, torch_mask = transformed_subject['volume'].data.squeeze(0), transformed_subject['mask'].data.squeeze(0)
            
            tmp_masks = []
            for i in range(one_hot_masks.shape[0]):
                key = f'one_hot_label_{i}'  
                one_hot_channel = transformed_subject[key].data.squeeze(0)
                tmp_masks.append(one_hot_channel)
                
            one_hot_masks = torch.stack(tmp_masks, dim=0)

        return torch_image.unsqueeze(0), torch_mask.unsqueeze(0), one_hot_masks.unsqueeze(1)
    
    def one_hot_encode(self, mask):
        unique_vals = self.index_classes

        one_hot = torch.zeros((len(unique_vals), *mask.shape), dtype=torch.float32)

        one_hot_map = {}
        for idx, val in enumerate(unique_vals):
            one_channel = (mask == val).to(torch.float32)
            one_hot[idx] = one_channel
            one_hot_map[idx] = [val, self.CLASSES[val]]

        return one_hot, one_hot_map

    @lru_cache(maxsize=1)
    def freesurfer_colormap(self, cmap):
        if cmap == 'lut':
            colordict = self.COLORDICT
        rgb_values = [(0, 0, 0) for _ in range(2036)]

        for key, value in colordict.items():
            if key < 2036:
                rgb_values[key] = value

        normalized_rgb_values = [(r/255, g/255, b/255) for r, g, b in rgb_values]

        custom_colormap = mcolors.ListedColormap(normalized_rgb_values)

        return custom_colormap

    def display_mpr(self, volume, cmap='gray'):
        def pad(img, target_size):
            pad = np.array(target_size) - np.array(img.shape)
            img = np.pad(img, [(pad[0]//2, pad[0] - pad[0]//2), (pad[1]//2, pad[1]-pad[1]//2)])
            return img

        c = np.array(volume.shape)//2
        ax = volume[c[0],:,:]
        co = volume[:,c[1],:]
        sa = volume[:,:,c[2]+c[2]//3]

        h = np.max([ax.shape, co.shape, sa.shape], axis=0)[0]
        ax = pad(ax, [h, ax.shape[1]])
        co = pad(co, [h, co.shape[1]])
        sa = pad(sa, [h, sa.shape[1]])

        dis = np.hstack([ax,co,sa])
        
        
        fig, ax = plt.subplots(figsize=(15, 15))
        if cmap == 'lut':
            cmap = self.freesurfer_colormap(cmap)
            im = ax.imshow(dis, cmap=cmap, interpolation='nearest', vmin=0, vmax=2035)
        else:
            im = ax.imshow(dis, cmap=cmap)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()