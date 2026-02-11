from __future__ import print_function
import argparse
import datetime
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from torch.utils.data import DataLoader
import models.unet as unet
from utils.tools import *
from utils.config import * 
import glob
from natsort import natsorted
from dsets.GaainDataset import GaainDataset
from sklearn.metrics import precision_recall_curve, auc
import gc


class GaainEval:
    def __init__(self, args):
        self.cli_args = args
        args_dict = {
            **vars(args),
        }
        self.classes = VM100_1['classes']
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.cli_args = argparse.Namespace(**args_dict)
        self.model_path = os.path.join('models', self.cli_args.group, self.cli_args.project)
        self.model_pattern = f'{self.cli_args.group}_avg_*.state'
        self.full_pattern = os.path.join(self.model_path, self.model_pattern)
        self.full_list = natsorted(glob.glob(self.full_pattern))
        
        
        self.transforms_dict = {
            'resize': tio.Resize(
                self.cli_args.resolution
            ),
            **(
                {'zoom': tio.RandomAffine(
                    scales=(self.cli_args.zoom, self.cli_args.zoom),
                    image_interpolation='bspline',
                    p=1.0,
                )} if self.cli_args.zoom > 1 else {}
            ),
            **(
                {'znorm': tio.ZNormalization()} if self.cli_args.znorm else {}
            ),
        }
        
    
    
    
    
    
    
    
    def loadModel(self, full_path):
        segmentation_model = unet.UNet3D(
            in_channels=1,
            num_classes=6,
        )
        model_info_dict = torch.load(full_path)
        segmentation_model.load_state_dict(model_info_dict['model_state'])
    
        if self.use_cuda:
            segmentation_model = segmentation_model.to(self.device)
        else:
            pass
            
        return segmentation_model
    
    
    
    def initValDl(self):
        test_ds = GaainDataset(
            patient_dir=VM100_1['test_dir'],
            images_dir=VM100_1['orig_file'],
            masks_dir=VM100_1['mask_file'],
            classes=self.classes,
            augmentation=tio.Compose([
                *[val for val in self.transforms_dict.values()]
            ]),
        )
        batch_size = self.cli_args.batch_size

        if os.name == 'nt':
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                drop_last=True,
                shuffle=False,
            )
        else:
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                drop_last=True,
                shuffle=False,
                worker_init_fn=lambda worker_id: seed_worker(worker_id, self.cli_args.seed)
            )
        return test_loader
    
    
    

    def calculate_precision_recall(self, prd, trg):
        prd_flat = prd.flatten()
        trg_flat = trg.flatten()

        precision, recall, thresholds = precision_recall_curve(trg_flat, prd_flat)
        return precision, recall, thresholds
    
    
    
    def main(self):
        dloader = self.initValDl()
        n_models = len(self.full_list)
        n_classes = len(self.classes)
        bucket_array = np.zeros((n_models, n_classes + 1))
        bucket_list = []
        for model_idx, el in enumerate(self.full_list):
            model = self.loadModel(el)
            model.eval()
            tmp = []
            with torch.no_grad():
                for bat_idx, batch_tup in enumerate(dloader):
                    if bat_idx > 0:
                        break
                    input_t, label_t, labels_t = batch_tup
                    input_g = input_t.to(self.device, non_blocking=True)
                    pred_g, _ = model(input_g)
                    pred_g = torch.sigmoid(pred_g.unsqueeze(2))

                    labels_n = labels_t.squeeze(2).cpu().numpy()
                    pred_n = pred_g.squeeze(2).cpu().detach().numpy()
                    ap_values = []
                    for idx in range(len(self.classes)):
                        trg = labels_n[:,idx]
                        prd = pred_n[:,idx]

                        precision, recall, thresholds = self.calculate_precision_recall(prd, trg)

                        ap = auc(recall, precision)
                        ap_values.append(ap)
                    mAP = np.mean(ap_values)
                    bucket_array[model_idx, :] += np.array([*ap_values, mAP])
                    tmp = [
                        os.path.basename(el),
                        f"{ap_values[0]:.3f}",
                        f"{ap_values[1]:.3f}",
                        f"{ap_values[2]:.3f}",
                        f"{ap_values[3]:.3f}",
                        f"{ap_values[4]:.3f}",
                        f"{ap_values[5]:.3f}",
                        f"{mAP:.3f}",
                    ]
                    print(tmp)
            bucket_list.append(tmp)

        header = ','.join([*self.classes, 'mAP'])
        np.savetxt(f"results/{self.cli_args.project}.csv", bucket_array, delimiter=",", fmt='%f', header=header, comments='')
        




                
        
        

if __name__ == '__main__':
    ctime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    try:
        parser = argparse.ArgumentParser(description='Run GaainFed locally')
        parser.add_argument('-GRP','--group',         default='none_group',                 help="Group folder for seperator.", )
        parser.add_argument('-PRJ','--project',       default='none_project',               help="Project folder for seperator.", )
        parser.add_argument('-SVR','--server',        default='avg',                help="Server name for seperator.", )
        parser.add_argument('-CNT', '--counter',     default=0,       type=int,            help='Select initial segmentation model.')
        parser.add_argument('-EPC', '--epochs',        default=1,       type=int,            help='Number of epochs to train for', )
        parser.add_argument('-BAT', '--batch_size', default=1, type=int, help='Batch size to use for training', )
        parser.add_argument('-NWK', '--num_workers',   default=4,       type=int,            help='Number of worker processes for background data loading', )
        parser.add_argument('comment',                default='none',  nargs='?',           help="Comment suffix for Tensorboard run.", )
        
        parser.add_argument('-RS','--resolution',      default=128,     type=int,            help='Pixel size to use for training', )
        parser.add_argument('-ZM','--zoom',            default=1.0,     type=float,          help="Augment the training data by zoom", )
        parser.add_argument('-ZN','--znorm',           default=True,    action='store_true', help="Augment the training data by znorm", )
        
        cli_args = parser.parse_args(sys.argv[1:])
        GaainEval(cli_args).main()
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise