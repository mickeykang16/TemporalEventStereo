from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *
from dataloader.dsecProvider_test import DatasetProvider
from dataloader.contrast.raw_event_utils import custom_collate

from utils.evaluation import do_evaluation
from utils.flow import EventWarping

import argparse
import os
import numpy as np
import time
from tqdm.auto import tqdm
from shutil import copyfile
import yaml
import cv2
import matplotlib.pyplot as plt
import warnings
import pdb
import threading
import flow_vis
import pathlib
from pathlib import Path
from collections import OrderedDict

class EventStereo():
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.cuda = torch.cuda.is_available()
        torch.manual_seed(args.seed)    
        if self.cuda:   
            torch.cuda.manual_seed(args.seed)
        self.epoch = int(config['model']['epoch'])
        self.frame_idxs = config['model']['frame_idxs']
        self.use_prev_gradient = config['model']['use_prev_gradient']
        
        if isinstance(self.frame_idxs, list):
            self.frame_idxs.sort()
        # Manage expression for frame_idxs
        elif isinstance(self.frame_idxs, str):
            self.frame_idxs = eval(self.frame_idxs)
            # check iterable
            assert len(self.frame_idxs) > 0
            for i in self.frame_idxs:
                assert isinstance(i, int)
            
        # Make dataloaders
        data_root = config['model']['dataset_root_path']
        height = config['model']['height']
        width = config['model']['width']
        in_ch = config['model']['in_ch']
        
        self.orig_height = config['model']['orig_height']
        self.orig_width = config['model']['orig_width']
        maxdisp = config['model']['maxdisp']
        self.eval_maxdisp = config['model']['eval_maxdisp']
        
        self.use_pseudo_gt = config['model'].get('use_pseudo_gt', False)
        self.use_disp_gt_mask = config['model'].get('use_disp_gt_mask', False)
        self.use_mini_data = config['model'].get('use_mini_data', False)
        self.use_super_mini_data = config['model'].get('use_super_mini_data', False)
        
        self.val_of_viz = config['model'].get('val_of_viz', False)

        
        # if self.use_pseudo_gt:
        pseudo_root_path = config['model']['pseudo_root_path']
        
        
        self.use_raw_provider = config['model'].get('use_raw_provider', False)
        
        data_root = config['model']['dataset_root_path']
        raw_data_root = config['model']['dataset_raw_root_path']
        
        self.use_disp_flow_warp_mask =  config['model'].get('use_disp_flow_warp_mask', False)
        self.use_featuremetric_loss = config['model'].get('use_featuremetric_loss', False)
        self.use_disp_loss = config['model'].get('use_disp_loss', True)
        self.use_contrast_loss = config['model'].get('use_contrast_loss',False)
        self.use_stereo_loss = config['model'].get('use_stereo_loss', True)
        
        
        dataset_provider = DatasetProvider(Path(data_root), Path(raw_data_root), frame_idxs = self.frame_idxs, eval_maxdisp=self.eval_maxdisp, img_load=True,
                                        num_bins = in_ch, pad_width=width, pad_height=height,
                                        pseudo_path= pseudo_root_path, use_mini = self.use_mini_data, use_super_mini = self.use_super_mini_data)
        self.test_dataset = dataset_provider.get_test_dataset()
        
        self.test_dataset[0]
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                        batch_size= config['test']['batch_size'],
                                                        shuffle= config['test']['shuffle'],
                                                        num_workers= config['test']['num_worker'],
                                                        drop_last=False, pin_memory=True)
        

        # define model
        model_name = config['model']['type']
        self.model_name = model_name
        
        if model_name == 'ours_large':
            self.model = ours_large(maxdisp, self.orig_height, self.orig_width, in_ch=in_ch)
        
        else:
            raise Exception('Wrong model type')
        
        self.L1_lossFn = nn.L1Loss()
        
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

        # add logger
        if 'debug' in args.savemodel:
            os.makedirs(args.savemodel, exist_ok=True)
        else:
            os.makedirs(args.savemodel, exist_ok=False)
        os.makedirs(os.path.join(args.savemodel, 'prediction'), exist_ok=True)
        os.makedirs(os.path.join(args.savemodel, 'warped'), exist_ok=True)
        os.makedirs(os.path.join(args.savemodel, 'flow'), exist_ok=True)
        copyfile(args.config, os.path.join(args.savemodel, 'config.yaml'))
        model_file_path = os.path.join('models', config['model']['type']+'.py')
        if os.path.isfile(model_file_path):
            copyfile(model_file_path, os.path.join(args.savemodel, config['model']['type']+'.py'))
        self.logger = Logger(args.savemodel)
        self.logger.log_and_print("Model name: " + model_name)
        self.logger.log_and_print("Log path: " + args.savemodel + "\n")
        tensorboard_path = os.path.join(args.savemodel, 'runs')
        self.writer = SummaryWriter(tensorboard_path)
        self.log_train_every_n_batch = int(config['log']['log_train_every_n_batch'])
        self.save_test_every_n_batch = int(config['log']['save_test_every_n_batch'])
        lr = self.config['lr']
        
        do_not_load_layer = config['model'].get('do_not_load_layer', [])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        loadmodel = config['model']['load_model']
        load_strict = config['model']['load_strict']
        load_optim = config['model']['load_optim']
        
        if loadmodel != '':
            if os.path.isfile(loadmodel):
                self.logger.log_and_print('Load pretrained model from '+ loadmodel)
                pretrain_dict = torch.load(loadmodel)
                    
                new_state_dict = OrderedDict()
                for k, v in pretrain_dict['state_dict'].items():
                    name = k.split('.')[1]
                    if name not in do_not_load_layer:
                        new_name = k
                        new_state_dict[new_name] = v
                    else:
                        print(k)
                self.model.load_state_dict(new_state_dict, strict=load_strict)
                
                if ('optimizer' in pretrain_dict) & load_optim:
                    self.optimizer.load_state_dict(pretrain_dict['optimizer'])
                else:
                    warnings.warn("Warning: Load state dict does not contain optimizer state, might cause different result!!!")
                
            else:
                raise Exception('Connot find pretrained file')
            
        self.logger.log_and_print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        self.logger.log_and_print('Done initialization\n')
        
    def remove_padding(self, image, pad):
        image_is_tensor = isinstance(image, torch.Tensor)
        if image_is_tensor:
            image = [image]
        pad_left, pad_right, pad_top, pad_bottom = pad
        pad_left = pad_left[0]
        pad_right = pad_right[0]
        pad_top = pad_top[0]
        pad_bottom = pad_bottom[0]
        full_h, full_w = image[0].shape[-2:]
        h = full_h - pad_top - pad_bottom
        w = full_w - pad_left - pad_right
        disp_no_pad = []
        for disp in image:
            assert disp.shape[-2] == full_h and disp.shape[-1] == full_w
            disp_no_pad.append(disp[... , pad_top:full_h - pad_bottom, pad_left:full_w - pad_right])
        
        if image_is_tensor:
            return disp_no_pad[0]    
        else:
            return disp_no_pad
        
    def calculate_contrast_loss(self, flow, e_list, pol_mask):
        loss = 0.0
        device = flow.device
        for b in range(flow.shape[0]):
            batch_flow = [flow[b].unsqueeze(0)]
            loss += self.contrast_lossFn(batch_flow, 
                                e_list[b].unsqueeze(0).to(device), 
                                pol_mask[b].unsqueeze(0).to(device))
        return loss

    def test(self, batch):
        self.model.eval()
        debug_batch = []
        img_batch = []
        disp_true_batch = []
        debug = {}
        total_batch_time = 0.0
        for i, (imgL, imgR, pad, data_debug, save_path) in enumerate(batch):
            frame_idx = self.frame_idxs[i]

            if self.cuda:
                imgL, imgR = imgL.cuda(), imgR.cuda()
                
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation 
            starter.record()
            with torch.no_grad():
                if self.model_name in ['ours_large']:
                    if i == 0:
                        prev_feat = None
                        prev_pred = None
                        prev_cost = None
                        prev_disp = None
                    output3, prev_feat, debug, cost_volume = self.model(imgL,imgR, prev_feat, prev_disp, prev_cost, prev_pred)
                    prev_pred = output3
                    small_disp_true = F.interpolate(output3, scale_factor = 0.25, mode="bilinear", align_corners=True)/4.0
                    prev_cost = cost_volume
                    debug['prev_gt'] = prev_disp
                    prev_disp = None
                output3 = torch.squeeze(output3, dim=1)
            ender.record()
            torch.cuda.synchronize()
            total_batch_time += starter.elapsed_time(ender)

        output_img = self.remove_padding([output3], pad)[0]
        
        save_name = os.path.join(self.args.savemodel, 'test', save_path[0])
        save_dir = os.path.join(self.args.savemodel, 'test', save_path[0][:-11])
        
        os.makedirs(save_dir, exist_ok=True)
        
        output_img = output_img.cpu()
        for i in range(output_img.shape[0]):
            img = output_img[i]
            img = np.array(img * 256, dtype = np.uint16)
            cv2.imwrite(str(save_name), img)
        
        # NOTE: UPDATED: for now img and gt disp of all samples are returned
        # img_batch and disp_true_batch have dimension of [num_frame_idxs x (B x H x W)]
        # debug_batch [num_frame_idxs x {"left_of", "right_of", "disp_of", ...}]
        return (total_batch_time/len(batch), debug_batch)

            
    def run(self):
        start_full_time = time.time()
        validation_result = []
        
        
        total_val_loss = 0
        disp_pred = []
        disp_gt = []
        total_batch_time = 0
        total_idx = 0
        total_photo_loss = 0.0 
        for batch_idx, batch in tqdm(enumerate(self.test_loader), 
                                        total=len(self.test_loader), 
                                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            assert len(batch) == len(self.frame_idxs)

            batch_time, debug = self.test(batch)
            
            total_batch_time += batch_time
            total_idx += 1
            
        self.logger.log_and_print('Average test time per batch = {:.3f}'.format(total_batch_time/total_idx))
    
def main():
    parser = argparse.ArgumentParser(description='TESNet')
    parser.add_argument('--config', default='',
                        help='config file path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--savemodel', default='./',
                    help='save model')
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    eventstereo = EventStereo(config=config, args=args)
    eventstereo.run()

if __name__ == '__main__':
   main()