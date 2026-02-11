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
import models.unet as unet
from utils.tools import *
from utils.config import * 
from utils.logconf import logging
import glob
import os

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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

class GaainFed:
    
    def __init__(self, args):
        self.cli_args = args
        self.time_str = ctime
        
        assert self.cli_args.counter > 0
        
        fix_seed(self.cli_args.seed, False)
        
            
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

        self.num_classes = self.analysis_dict['classes'].__len__()
        self.segmentation_model = self.initModel()


    def initModel(self):
        segmentation_model = unet.UNet3D(
            in_channels=1,
            num_classes=self.num_classes,
        )
        if self.cli_args.counter == 0:
            raise ValueError(f"잘못된 counter 값 입력: {self.cli_args.counter} (must counter > 0)")
        elif self.cli_args.counter == 1:
            pass
        else:
            _path = os.path.join('models', self.cli_args.group, self.cli_args.project)
            previous_counter = self.cli_args.counter - 1
            _key = f"{self.cli_args.group}_*_{previous_counter}.state"
            pattern = os.path.join(_path, _key)
            statelist = glob.glob(pattern)
            print(statelist)
            assert len(statelist) != 0

            for layer_name in segmentation_model.state_dict():
                if "num_batches_tracked" in layer_name:
                    continue
                segmentation_model.state_dict()[layer_name] *= 0
            
            for idx, el in enumerate(statelist):
                _state = torch.load(el, map_location=torch.device('cpu'))['model_state']
                for layer_name in segmentation_model.state_dict():
                    if "num_batches_tracked" in layer_name:
                        continue
                    _state[layer_name] *= (1/len(statelist))
                    segmentation_model.state_dict()[layer_name] += _state[layer_name]


            
        return segmentation_model


    def saveModel(self, group, project, server, epoch_ndx, isBest=False):
        if self.cli_args.counter == 0:
            return None
        elif self.cli_args.counter == 1:
          file_path = os.path.join(
              'models',
              group,
              project,
              f'init_model.state'
          )
        else:
          file_path = os.path.join(
              'models',
              group,
              project,
              f'{group}_{server}_{self.cli_args.counter}.state'
          )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        model = self.segmentation_model
        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
        }
        torch.save(state, file_path)
        return None

    def main(self):
        self.saveModel(self.cli_args.group, self.cli_args.project, self.cli_args.server, self.cli_args.epochs, False)


if __name__ == '__main__':
    ctime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    try:
        parser = argparse.ArgumentParser(description='Run GaainFed locally')
        parser.add_argument('-CNT', '--counter',     default=0,       type=int,            help='Select initial segmentation model.')
        parser.add_argument('-ANL', '--analysis',     default=1,       type=int,            help='(1) basic', )
        parser.add_argument('-S',   '--seed',         default=1,       type=int,            help='random seed (default: 1)', )
        parser.add_argument('-EPC', '--epochs',        default=1,       type=int,            help='Number of epochs to train for', )

        parser.add_argument('-GRP','--group',         default='none_group',                 help="Group folder for seperator.", )
        parser.add_argument('-PRJ','--project',       default='none_project',               help="Project folder for seperator.", )
        parser.add_argument('-SVR','--server',        default='avg',                help="Server name for seperator.", )
        parser.add_argument('comment',                default='none',  nargs='?',           help="Comment suffix for Tensorboard run.", )
        cli_args = parser.parse_args(sys.argv[1:])
        GaainFed(cli_args).main()
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise