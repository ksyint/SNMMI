import argparse
import datetime
import os
import shutil
import copy
import sys
import signal
import atexit
import numpy as np
import json
import torch
import torch.nn as nn
import torchio as tio
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import models.unet as unet
import utils.logconf as logconf
from dsets.GaainDataset import GaainDataset
from utils.logconf import logging
from utils.tools import *
from utils.config import * 
from utils.tools import enumerateWithEstimate


main_pid = os.getpid()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    


def create_histogram_landmarks(reference_image_paths, output_path='utils/landmarks.json'):
    landmarks = tio.HistogramStandardization.train(reference_image_paths)
    
    landmarks_list = landmarks.tolist()

    with open(output_path, 'w') as f:
        json.dump(landmarks_list, f)
    return landmarks

def fix_seed(random_seed, use_cuda):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class GaainApp:
    NORM_COLORDICT = {k: [v_i / 255.0 for v_i in v] for k, v in COLORDICT.items()}
    
    
    def __init__(self, args):
        self.selected_gpu_count = 1
        self.cli_args = args
        self.time_str = ctime
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            if self.cli_args.gpus is not None:
                gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        analaysis_list = [
            VM100_1,
            VM100_2,
            VP100_1,
            VP100_2,
            VV100_1,
            VV100_2,
        ]
        
        if self.cli_args.analysis < len(analaysis_list):
            self.analysis_dict = analaysis_list[self.cli_args.analysis]
        else:
            self.analysis_dict = analaysis_list[0]


        self.cli_args.num_workers = self.analysis_dict['num_workers']
        self.cli_args.batch_size = self.analysis_dict['batch_size']
        self.cli_args.seed = self.analysis_dict['seed']
        self.cli_args.lr = self.analysis_dict['lr']
        self.cli_args.scheduler_step = self.analysis_dict['scheduler_step']
        self.cli_args.scheduler_gamma = self.analysis_dict['scheduler_gamma']
        self.cli_args.w_bce = self.analysis_dict['w_bce']
        self.cli_args.w_dice = self.analysis_dict['w_dice']
        self.cli_args.w_l1 = self.analysis_dict['w_l1']
        self.cli_args.w_penalty = self.analysis_dict['w_penalty']
        self.cli_args.epsilon = self.analysis_dict['epsilon']
        self.cli_args.preprocess = self.analysis_dict['preprocess']
        self.cli_args.resolution = self.analysis_dict['resolution']
        self.cli_args.strategy = self.analysis_dict['strategy']
        self.cli_args.canonical = self.analysis_dict['canonical']
        self.cli_args.deformation = self.analysis_dict['deformation']
        self.cli_args.biasfield = self.analysis_dict['biasfield']
        self.cli_args.noise = self.analysis_dict['noise']
        self.cli_args.flip = self.analysis_dict['flip']
        self.cli_args.affine = self.analysis_dict['affine']
        self.cli_args.zoom = self.analysis_dict['zoom']
        self.cli_args.znorm = self.analysis_dict['znorm']
        
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cli_args.seed)

        fix_seed(self.cli_args.seed, self.use_cuda)

        self.ref_t_writer = None
        self.ref_v_writer = None
        self.trn_writer = None
        self.val_writer = None
        self.initTensorboardWriters()

        self.train_dir = self.analysis_dict['train_dir']
        self.test_dir = self.analysis_dict['test_dir']
        self.orig_file = self.analysis_dict['orig_file']
        self.mask_file = self.analysis_dict['mask_file']
        self.classes = self.analysis_dict['classes']
        self.num_classes = self.analysis_dict['classes'].__len__()
        self.weights = self.analysis_dict['weights']
        self.mu = self.analysis_dict['mu']
        self.bce_loss = torch.nn.BCELoss()
        self.l1_loss = torch.nn.L1Loss()

        flipped = {value: key for key, value in CLASSES.items()}
        self.index_classes = sorted([flipped[el] for el in self.classes])

        self.preprocessing_list = ['skullstriping'] if self.cli_args.preprocess else None
            
        self.transforms_dict = {
            **(
                {'zoom': tio.RandomAffine(
                    scales=(1.0, self.cli_args.zoom),
                    degrees=0,
                    translation=(0, 0, 0),
                    image_interpolation='bspline',
                    p=1.0,
                )} if self.cli_args.zoom > 1 else {}
            ),
            **(
                {'resize': tio.Resize(
                        self.cli_args.resolution
                )} if True else {}
            ),
            **(
                {'canonical': tio.ToCanonical(),
                } if self.cli_args.canonical else {}
            ),
            **(
                {'deformation': tio.RandomElasticDeformation(
                    num_control_points=(5, 5, 5),
                    max_displacement=(3, 3, 3),
                    p=self.cli_args.deformation,
                )} if self.cli_args.deformation > 0 else {}
            ),
            **(
                {'biasfield': tio.RandomBiasField(
                    p=self.cli_args.biasfield
                )} if self.cli_args.biasfield > 0 else {}
            ),
            **(
                {'noise': tio.RandomNoise(
                    p=self.cli_args.noise
                )} if self.cli_args.noise > 0 else {}
            ),
            **(
                {'flip': tio.RandomFlip(
                    axes=(0, 1, 2), 
                    p=self.cli_args.flip
                )} if self.cli_args.flip > 0 else {}
            ),
            **(
                {'affine': tio.RandomAffine(
                    degrees=(3, 3, 3),
                    translation=(0, 0, 0),
                    image_interpolation='bspline',
                    p=self.cli_args.affine,
                )} if self.cli_args.affine > 0 else {}
            ),
            **(
                {'znorm': tio.ZNormalization()} if self.cli_args.znorm else {}
            ),
        }

        self.transforms_dict_test = {
            **(
                {'zoom': tio.RandomAffine(
                    scales=(self.cli_args.zoom, self.cli_args.zoom),
                    image_interpolation='bspline',
                    p=1.0,
                )} if self.cli_args.zoom > 1 else {}
            ),
            **(
                {'resize': tio.Resize(
                        self.cli_args.resolution
                )} if True else {}
            ),
            **(
                {'canonical': tio.ToCanonical(),
                } if self.cli_args.canonical else {}
            ),
            **(
                {'znorm': tio.ZNormalization()} if self.cli_args.znorm else {}
            ),
        }
        
        self.segmentation_model = self.initModel()
        if self.cli_args.strategy == 'fedprox':
            self.segmentation_model_global = copy.deepcopy(self.segmentation_model)
            log.info(f"Using Strategy; {self.cli_args.strategy}:{self.mu} (fedprox:mu).")
        else:
            self.segmentation_model_global = None
            log.info(f"Using Strategy; {self.cli_args.strategy} (fedavg).")
        self.optimizer = self.initOptimizer()
        self.scheduler = self.initScheduler(self.optimizer,
                                            self.cli_args.scheduler_step,
                                            self.cli_args.scheduler_gamma
                                            )

    def initTrainDl(self):
        train_ds = GaainDataset(
            patient_dir=self.train_dir,
            images_dir=self.orig_file,
            masks_dir=self.mask_file,
            classes=self.classes,
            preprocessing=self.preprocessing_list,
            augmentation=tio.Compose([
                *[val for val in self.transforms_dict.values()]
            ]),
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= self.selected_gpu_count

        if os.name == 'nt':
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                drop_last=True,
                shuffle=True
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                drop_last=True,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.generator
            )
        return train_loader

    def initValDl(self):
        test_ds = GaainDataset(
            patient_dir=self.test_dir,
            images_dir=self.orig_file,
            masks_dir=self.mask_file,
            classes=self.classes,
            preprocessing=self.preprocessing_list,
            augmentation=tio.Compose([
                *[val for val in self.transforms_dict_test.values()]
            ]),
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= self.selected_gpu_count

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
                worker_init_fn=seed_worker,
                generator=self.generator
            )
        return test_loader

    def initScheduler(self, optimizer, step_size=1, gamma=1):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    def initOptimizer(self):
        return torch.optim.Adam(self.segmentation_model.parameters(), lr=self.cli_args.lr)

    def calculate_proximal_term(self, local_model_parameters, global_model_parameters):
        proximal_term = 0.0
        for local_param, global_param in zip(local_model_parameters, global_model_parameters):
            proximal_term += ((local_param - global_param) ** 2).sum()
        return proximal_term

    def multi_label_dice_loss(self, predicted, targets, num_classes, weights=None):
        """Calculate Dice loss for multiple labels."""
        if weights is None:
            while True:
                weights = torch.randint(0, 2, (num_classes,))
                if weights.sum() >= 1:
                    weights.tolist()
                    break

        total_loss = 0.
        labels_losses = torch.zeros(num_classes, targets.shape[0], device=self.device)
        for i in range(num_classes):
            dice_loss_for_class = self.dice_loss(predicted[:, i], targets[:, i], epsilon=self.cli_args.epsilon)
            labels_losses[i, :] = dice_loss_for_class
            total_loss += weights[i] * dice_loss_for_class
        return total_loss / sum(weights), labels_losses

    def multi_label_bce_loss(self, predicted, targets, num_classes):
        total_loss = 0.0
        for i in range(num_classes):
            prediction_slice = predicted[:, i]
            target_slice = targets[:, i]
            loss = self.bce_loss(prediction_slice, target_slice)
            total_loss += loss
        return total_loss / num_classes

    def multi_label_l1_loss(self, predicted, targets, num_classes):
        total_loss = 0.0
        for i in range(num_classes):
            prediction_slice = predicted[:, i]
            target_slice = targets[:, i]
            loss = self.l1_loss(prediction_slice, target_slice)
            total_loss += loss
        return total_loss / num_classes

    def dice_loss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=tuple(range(1, label_g.dim())))
        dicePrediction_g = prediction_g.sum(dim=tuple(range(1, prediction_g.dim())))
        diceCorrect_g = (prediction_g * label_g).sum(dim=tuple(range(1, label_g.dim())))

        diceRatio_g = (2 * diceCorrect_g) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g
    
    def initWeights(self):
        if self.cli_args.counter == 0:
            self.cli_args.counter = 1
        elif self.cli_args.counter == 1:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, 'init_model.state')
            init_unet_dict = torch.load(path)
        else:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, f'{self.cli_args.group}_avg_{self.cli_args.counter}.state')
            init_unet_dict = torch.load(path)
            segmentation_model.load_state_dict(init_unet_dict['losses'])
            

    def initModel(self):
        segmentation_model = unet.UNet3D(
            in_channels=1,
            num_classes=self.num_classes,
        )

        if self.cli_args.counter == 0:
            self.cli_args.counter = 1
        elif self.cli_args.counter == 1:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, 'init_model.state')
            init_unet_dict = torch.load(path)
            segmentation_model.load_state_dict(init_unet_dict['model_state'])
        else:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, f'{self.cli_args.group}_avg_{self.cli_args.counter}.state')
            init_unet_dict = torch.load(path)
            segmentation_model.load_state_dict(init_unet_dict['model_state'])

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                if self.cli_args.gpus is not None:
                    gpu_ids = [int(gpu_id.strip()) for gpu_id in self.cli_args.gpus.split(',')]
                    self.selected_gpu_count = len(gpu_ids)
                    segmentation_model = nn.DataParallel(segmentation_model, device_ids=gpu_ids)
                else:
                    segmentation_model = nn.DataParallel(segmentation_model)
                    self.selected_gpu_count = torch.cuda.device_count()
                segmentation_model = segmentation_model.to(self.device)
            else:
                segmentation_model = segmentation_model.to(self.device)
                self.selected_gpu_count = 1
            log.info(f"Using CUDA; {self.selected_gpu_count} devices.")
        return segmentation_model

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(OFFSET_METRIC + self.num_classes, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                loss_var, latent_vectors, labels = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    random_weight=False,
                )


        return valMetrics_g.to('cpu')

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(OFFSET_METRIC + self.num_classes, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var, _, _ = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                random_weight=False,
            )
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size,
                         metrics_g,
                         random_weight=False,
                         classificationThreshold=0.5,
                         ):
        w_bce, w_dice, w_l1, w_penalty = self.cli_args.w_bce, self.cli_args.w_dice, self.cli_args.w_l1, self.cli_args.w_penalty
        
        input_t, label_t, labels_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        labels_g = labels_t.to(self.device, non_blocking=True)


        prediction_g, latent_variables = self.segmentation_model(input_g)

        prediction_g = prediction_g.unsqueeze(2)
        prediction_g = torch.sigmoid(prediction_g)
        bceLoss_g = self.multi_label_bce_loss(prediction_g, labels_g, self.num_classes) if w_bce > 0. else 0
        
        if random_weight:
            weights = None
        else:
            weights = self.weights
        
        diceLoss_g, labels_losses = self.multi_label_dice_loss(prediction_g, labels_g, self.num_classes, weights)
        l1Loss_g = self.multi_label_l1_loss(prediction_g, labels_g, self.num_classes) if w_l1 > 0. else 0
        overlap_penalty = 0


        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        combinedLoss_g = w_bce * bceLoss_g + w_dice * diceLoss_g.mean() + w_l1 * l1Loss_g + overlap_penalty

        if self.cli_args.strategy == 'fedprox':
            global_model_parameters = [param.data for param in self.segmentation_model_global.parameters()]
            local_model_parameters = [param.data for param in self.segmentation_model.parameters()]
            proximal_term = self.calculate_proximal_term(local_model_parameters, global_model_parameters)
            mu = self.mu
            log.info(f"Fedprox loss: {combinedLoss_g + mu * proximal_term} = {combinedLoss_g}+{mu}*{proximal_term} (fedprox:combinedLoss_g + mu * proximal_term).")
            combinedLoss_g = combinedLoss_g + mu * proximal_term
        else:
            log.info(f"Fedavg loss: combinedLoss_g = {combinedLoss_g} (fedavg:combinedLoss_g).")
            

        with torch.no_grad():
            metrics_g[0, start_ndx:end_ndx] = diceLoss_g
            for idx in range(0, self.num_classes):
                metrics_g[OFFSET_METRIC + idx, start_ndx:end_ndx] = labels_losses[idx, :]

        return combinedLoss_g, latent_variables, label_t

    def main(self):
        log.info(f"Starting {type(self).__name__}: {self.analysis_dict['description']}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_loss = 1.0

        epoch_ndx = 0 
        epoch_with_cycle = self.cli_args.epochs * (self.cli_args.counter - 1) + epoch_ndx - self.cli_args.offset

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            epoch_with_cycle = self.cli_args.epochs * (self.cli_args.counter - 1) + epoch_ndx - self.cli_args.offset
            log.info(f"Epoch {epoch_ndx}, LR={self.scheduler.get_last_lr()[0]} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} batches of size {self.cli_args.batch_size}*{(self.selected_gpu_count if self.use_cuda else 1)}")
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_with_cycle, 'trn', trnMetrics_t, self.trn_writer)

            if (epoch_ndx == 1) or (epoch_ndx == self.cli_args.epochs) or (epoch_ndx % self.cli_args.validation_cadence == 0):
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                mean_loss = self.logMetrics(epoch_with_cycle, 'val', valMetrics_t, self.val_writer)
                self.saveModel(self.cli_args.group, self.cli_args.project, self.cli_args.server, epoch_ndx, valMetrics_t, mean_loss == best_loss)
                best_loss = min(mean_loss, best_loss)

                if epoch_with_cycle == 1:
                    self.logImages(epoch_with_cycle, '1_1_ref_trn', train_dl, 0, self.ref_t_writer)
                    self.logImages(epoch_with_cycle, '2_1_ref_val', val_dl, 1, self.ref_v_writer)
                self.logImages(epoch_with_cycle, '1_2_prd_trn', train_dl, 0, self.trn_writer)
                self.logImages(epoch_with_cycle, '1_2_th_trn', train_dl, 0, self.trn_writer)
                self.logImages(epoch_with_cycle, '2_2_prd_val', val_dl, 1, self.val_writer)
                self.logImages(epoch_with_cycle, '2_2_th_val', val_dl, 1, self.val_writer)
            self.scheduler.step()
        self.ref_t_writer.close()
        self.ref_v_writer.close()
        self.trn_writer.close()
        self.val_writer.close()

    def logImages(self, epoch_ndx, mode_str, dl, pid, writer):
        self.segmentation_model.eval()
        dat = dl.dataset[pid]
        vol = dat[0].unsqueeze(0).to(self.device, non_blocking=True)
        msk = dat[1].unsqueeze(0).to(self.device, non_blocking=True)
        p_num = dl.dataset.get_pid(pid)
        c = np.array(vol.shape[-3:]) // 2
        ax_ndx = c[0] - c[0] // 6
        co_ndx = c[1] + c[1] // 6
        sa_ndx_right = c[2] - c[2] // 5
        sa_ndx_left = c[2] + c[2] // 7

        pred_msks = self.segmentation_model(vol)[0][0, :].unsqueeze(1)
        dims = list(range(vol.dim()))
        dims[-3], dims[-2], dims[-1] = dims[-2], dims[-3], dims[-1]
        all_msk = torch.sigmoid(pred_msks.float())
        all_msk = all_msk.permute(*dims)
        msk = msk.permute(*dims)
        ax_msk = msk[0, :, ax_ndx, :, :].detach().to('cpu')
        co_msk = msk[0, :, :, co_ndx, :].detach().to('cpu')
        sa_msk_R = msk[0, :, :, :, sa_ndx_right].detach().to('cpu')
        sa_msk_L = msk[0, :, :, :, sa_ndx_left].detach().to('cpu')

        ax_msk_color = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in ax_msk.flatten()], dim=0).reshape(
            ax_msk.shape[-2], ax_msk.shape[-1], 3).permute(2, 0, 1)
        co_msk_color = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in co_msk.flatten()], dim=0).reshape(
            co_msk.shape[-2], co_msk.shape[-1], 3).permute(2, 0, 1)
        sa_msk_color_R = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in sa_msk_R.flatten()],
            dim=0).reshape(sa_msk_R.shape[-2], sa_msk_R.shape[-1], 3).permute(2, 0, 1)
        sa_msk_color_L = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in sa_msk_L.flatten()],
            dim=0).reshape(sa_msk_L.shape[-2], sa_msk_L.shape[-1], 3).permute(2, 0, 1)

        if mode_str in ['1_1_ref_trn', '2_1_ref_val']:
            vol = vol.permute(*dims)
            min_vol, max_vol = torch.min(vol).to('cpu'), torch.max(vol).to('cpu')
            ax_vol = vol[0, :, ax_ndx, :, :].detach().to('cpu')
            co_vol = vol[0, :, :, co_ndx, :].detach().to('cpu')
            sa_vol_R = vol[0, :, :, :, sa_ndx_right].detach().to('cpu')
            sa_vol_L = vol[0, :, :, :, sa_ndx_left].detach().to('cpu')
            reg_ax_vol = (ax_vol - min_vol) / (max_vol - min_vol)
            reg_co_vol = (co_vol - min_vol) / (max_vol - min_vol)
            reg_sa_vol_R = (sa_vol_R - min_vol) / (max_vol - min_vol)
            reg_sa_vol_L = (sa_vol_L - min_vol) / (max_vol - min_vol)
            writer.add_image(f'{mode_str}/{p_num}_img_ax_{ax_ndx}', reg_ax_vol, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_co_{co_ndx}', reg_co_vol, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_sa_R_{sa_ndx_right}', reg_sa_vol_R, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_sa_L_{sa_ndx_left}', reg_sa_vol_L, epoch_ndx)

            writer.add_image(f'{mode_str}/{p_num}_msk_ax_{ax_ndx}', ax_msk_color, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_co_{co_ndx}', co_msk_color, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_sa_R_{sa_ndx_right}', sa_msk_color_R, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_sa_L_{sa_ndx_left}', sa_msk_color_L, epoch_ndx)
        else:

            all_ax_msk = all_msk[:, :, ax_ndx, :, :]
            all_co_msk = all_msk[:, :, :, co_ndx, :]
            all_sa_msk_R = all_msk[:, :, :, :, sa_ndx_right]
            all_sa_msk_L = all_msk[:, :, :, :, sa_ndx_left]

            for idx in range(len(all_msk)):
                label_idx = self.index_classes[idx]
                label_name = CLASSES[label_idx]


                if mode_str in ['1_2_prd_trn', '2_2_prd_val']:
                    writer.add_image(f'{mode_str}/{p_num}_prd_ax_{label_idx}_{label_name}_{ax_ndx}', all_ax_msk[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_co_{label_idx}_{label_name}_{co_ndx}', all_co_msk[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_sa_R_{label_idx}_{label_name}_{sa_ndx_right}',all_sa_msk_R[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_sa_L_{label_idx}_{label_name}_{sa_ndx_left}',all_sa_msk_L[idx], epoch_ndx)
            if mode_str in ['1_2_th_trn', '2_2_th_val']:
                tmp = (torch.clamp(
                    all_msk.max(dim=0)[0], 
                    0, 
                    0.2
                )/0.2).cpu().detach().numpy()

                tmp_ax = tmp[:,ax_ndx,:,:].squeeze(0)
                tmp_co = tmp[:,:,co_ndx,:].squeeze(0)
                tmp_sa_R = tmp[:,:,:,sa_ndx_right].squeeze(0)
                tmp_sa_L = tmp[:,:,:,sa_ndx_left].squeeze(0)

                combined_mask_ax = self.combine_masks(ax_msk_color, tmp_ax)
                combined_mask_co = self.combine_masks(co_msk_color, tmp_co)
                combined_mask_sa_r = self.combine_masks(sa_msk_color_R, tmp_sa_R)
                combined_mask_sa_l = self.combine_masks(sa_msk_color_L, tmp_sa_L)

                writer.add_image(f'{mode_str}/{p_num}_ax_all',combined_mask_ax, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_co_all',combined_mask_co, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_sa_R_all',combined_mask_sa_r, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_sa_L_all',combined_mask_sa_l, epoch_ndx)
        writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, writer):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        assert np.isfinite(metrics_a).all()


        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[0].mean()
        for idx in range(self.num_classes):
            metrics_dict[f'loss/{self.index_classes[idx]}_{CLASSES[self.index_classes[idx]]}'] = metrics_a[OFFSET_METRIC + idx].mean()

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + f"{self.cli_args.counter} cycle, "
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))

        prefix_str = '0_1_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, epoch_ndx)

        writer.flush()


        return metrics_dict['loss/all']

    def pad(self, img, target_size):
        pad = np.array(target_size) - np.array(img.shape)
        img = np.pad(img, [(pad[0] // 2, pad[0] - pad[0] // 2), (pad[1] // 2, pad[1] - pad[1] // 2)])
        return img

    def combine_masks(self, target, prediction, alpha=0.5):
        prediction_color = np.stack((prediction,) * 3, axis=0)

        combined = (target * alpha + prediction_color * (1 - alpha))
        return combined

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.group, self.cli_args.project, self.cli_args.server)

            self.ref_t_writer = SummaryWriter(
                log_dir=log_dir + '_ref_trn')
            self.ref_v_writer = SummaryWriter(
                log_dir=log_dir + '_ref_val')
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_prd_trn')
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_prd_val')


    def saveModel(self, group, project, server, epoch_ndx, metrics_t, isBest=False):
        
        metrics_a = metrics_t.detach().numpy()
        assert np.isfinite(metrics_a).all()
        loss_list = []
        for idx in range(self.num_classes):
            loss_list.append(metrics_a[OFFSET_METRIC + idx].mean())

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        file_path = os.path.join(
            'models',
            group,
            project,
            f'{group}_{server}_{self.cli_args.counter}.state'
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        state = {
            'losses': loss_list,
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
        }

        torch.save(state, file_path)

        log.info(f"Saved model params to {file_path}")

        if isBest:
            best_path = os.path.join(
                'models',
                group,
                project,
                f'{group}_{server}_{type(model).__name__}.best.state'
            )
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))




OFFSET_METRIC = 1

if __name__ == '__main__':
    def exit_handler():
        if os.getpid() == main_pid:
            log.info(f"Main-Program: [PID:{main_pid}] is exiting...")
        else:
            log.info(f"Child-Workers: [PID:{os.getpid()}] is exiting...")


    ctime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    logconf.setup_logging(ctime)
    atexit.register(exit_handler)
    
    try:
        parser = argparse.ArgumentParser(description="Load model in GainApp class")
        parser.add_argument('-G',   '--gpus',         default=None,    type=str, help='Comma-separated list of GPU device IDs to use (e.g., "0,1" for using GPU 0 and 1), leave empty to use all available GPUs')
        parser.add_argument('-OFF', '--offset',      default=0,       type=int,            help='starting offset for epoch_cycle.')
        parser.add_argument('-CNT', '--counter',      default=0,       type=int,            help='Select initial segmentation model.')
        parser.add_argument('-ANL', '--analysis',      default=1,       type=int,            help='(1) basic', )
        parser.add_argument('-NWK', '--num_workers',   default=4,       type=int,            help='Number of worker processes for background data loading', )
        parser.add_argument('-VLC', '--validation_cadence',default=5,   type=int,            help='Number of epochs to save model and validation for', )
        parser.add_argument('-EPC', '--epochs',        default=1,       type=int,            help='Number of epochs to train for', )
        parser.add_argument('-BAT', '--batch_size', default=1, type=int, help='Batch size to use for training', )

        parser.add_argument('-S', '--seed',            default=1,       type=int,            help='random seed (default: 1)', )
        parser.add_argument('-LR', '--lr',             default=1e-3,    type=float,          help='lr value for Adam optimizer', )
        parser.add_argument('-SS', '--scheduler_step', default=1,       type=int,            help='scheduler step for optimizer', )
        parser.add_argument('-SG', '--scheduler_gamma',default=1.0,       type=float,          help='scheduler gamma for optimizer', )
        parser.add_argument('-B', '--w_bce',           default=0,       type=float,          help='weight for bce-loss', )
        parser.add_argument('-D', '--w_dice',          default=1,       type=float,          help='weight for dice-loss', )
        parser.add_argument('-L', '--w_l1',            default=0,       type=float,          help='weight for l1-loss', )
        parser.add_argument('-P', '--w_penalty',       default=0,       type=float,          help='weight for overlay penalty', )
        parser.add_argument('-E', '--epsilon',         default=1e-5,    type=float,          help='Epsilon value for Dice Loss', )

        parser.add_argument('-PR','--preprocess',      default=False,   action='store_true', help="Preprocessing all data by skullstriping", )
        parser.add_argument('-RS','--resolution',      default=128,     type=int,            help='Pixel size to use for training', )
        parser.add_argument('-CN','--canonical',       default=True,    action='store_true', help="Augment the training data by canonical", )
        parser.add_argument('-DF','--deformation',     default=0.0,     type=float,          help="Augment the training data by deformation", )
        parser.add_argument('-BF','--biasfield',       default=0.0,     type=float,          help="Augment the training data by biasfield", )
        parser.add_argument('-NS','--noise',           default=0.0,     type=float,          help="Augment the training data by noise", )
        parser.add_argument('-FL','--flip',            default=0.0,     type=float,          help="Augment the training data by flip", )
        parser.add_argument('-AF','--affine',          default=0.0,     type=float,          help="Augment the training data by affine", )
        parser.add_argument('-ZM','--zoom',            default=1.0,     type=float,          help="Augment the training data by zoom", )
        parser.add_argument('-ZN','--znorm',           default=True,    action='store_true', help="Augment the training data by znorm", )

        parser.add_argument('-GRP', '--group',         default='none_group',                 help="Group folder for seperator.", )
        parser.add_argument('-PRJ', '--project',       default='none_project',               help="Project folder for seperator.", )
        parser.add_argument('-STG', '--strategy',      default='fedavg',                     help="Select federation strategy. [fedavg, fedprox]", )
        parser.add_argument('-SVR', '--server',        default='none_server',                help="Server name for seperator.", )

        parser.add_argument('comment',                 default='none',  nargs='?',           help="Comment suffix for Tensorboard run.", )
        cli_args = parser.parse_args(sys.argv[1:])
        GaainApp(cli_args).main()
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise