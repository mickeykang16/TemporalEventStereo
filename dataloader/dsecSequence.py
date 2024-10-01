from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from .representations import VoxelGrid
from .eventslicer import EventSlicer
from .contrast.raw_event_utils import EventSequenceLoader
import PIL.Image
from . import common
import os
import yaml
import hdf5plugin
import pdb



class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=5, 
                 frame_idxs: list=range(-2,1),  eval_maxdisp = 192, img_load: bool=False,
                 pad_width = 648, pad_height = 480, pseudo_path=None, raw_seq_path = None):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        self.frame_idxs = frame_idxs
        self.img_load = img_load
        self.eval_maxdisp = eval_maxdisp

        # Save output dimensions
        self.seq_path = seq_path
        
        self.pseudo_path = pseudo_path
        self.raw_seq_path = raw_seq_path
        
        assert str(seq_path).split('/')[-1] == str(raw_seq_path).split('/')[-1]
        
        self.pad_width = pad_width
        self.pad_height = pad_height
        
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.image_height = 1080
        self.image_width = 1440
        #  
        event_sequence_loader_config={
            'resolution': [pad_height, pad_width],
        }
        self.event_sequence_loader = EventSequenceLoader(event_sequence_loader_config)
        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']
        
        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        if self.mode == 'test':
            seq_str = str(seq_path).split('/')[-1]
            
            load_timestamps = np.loadtxt(seq_path / (seq_str + '.csv'), delimiter=",", dtype='int64')
            self.timestamps_test = load_timestamps[:, 0]
            self.image_index = load_timestamps[:, 1]
            
            img_dir = seq_path / 'images'
            assert img_dir.is_dir()
            timestamps = np.loadtxt(img_dir / 'timestamps.txt', dtype='int64')

            timestamps = timestamps[::2]
            self.timestamps = timestamps[1:]
            
        else:
            # load disparity timestamps
            disp_dir = seq_path / 'disparity'
            assert disp_dir.is_dir()
            
            self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

            # load disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                disp_gt_pathstrings.append(str(entry))
            disp_gt_pathstrings.sort()
            self.disp_gt_pathstrings = disp_gt_pathstrings
            assert len(self.disp_gt_pathstrings) == self.timestamps.size

        if self.img_load:
            with open(seq_path / 'calibration/cam_to_cam.yaml') as f:
                self.conf = yaml.load(f, Loader=yaml.FullLoader)

            cam0_int = self.conf['intrinsics']['camRect0']['camera_matrix']
            cam1_int = self.conf['intrinsics']['camRect1']['camera_matrix']
            cam2_int = self.conf['intrinsics']['camRect2']['camera_matrix']
            cam3_int = self.conf['intrinsics']['camRect3']['camera_matrix']

            T32 = np.array(self.conf['extrinsics']['T_32'])
            T10 = np.array(self.conf['extrinsics']['T_10'])

            R_rect0 = np.array(self.conf['extrinsics']['R_rect0'])
            R_rect1 = np.array(self.conf['extrinsics']['R_rect1'])
            R_rect2 = np.array(self.conf['extrinsics']['R_rect2'])
            R_rect3 = np.array(self.conf['extrinsics']['R_rect3'])

            Kr0 = np.array([[cam0_int[0], 0, cam0_int[2]], 
                            [0, cam0_int[1], cam0_int[3]], 
                            [0, 0, 1]])
            Kr1 = np.array([[cam1_int[0], 0, cam1_int[2]], 
                            [0, cam1_int[1], cam1_int[3]], 
                            [0, 0, 1]])
            Kr2 = np.array([[cam2_int[0], 0, cam2_int[2]], 
                            [0, cam2_int[1], cam2_int[3]], 
                            [0, 0, 1]])
            Kr3 = np.array([[cam3_int[0], 0, cam3_int[2]], 
                            [0, cam3_int[1], cam3_int[3]], 
                            [0, 0, 1]])

            M1=np.matmul(Kr1,R_rect1)
            M2=np.matmul(M1,T10[:3,:3])
            M3=np.matmul(M2,np.linalg.inv(R_rect0))
            self.homography_left=np.matmul(M3,np.linalg.inv(Kr0))


            M1=np.matmul(Kr3,R_rect3)
            M2=np.matmul(M1,T32[:3,:3])
            M3=np.matmul(M2,np.linalg.inv(R_rect2))

            # M3 = Kr3
            self.homography_right=np.matmul(M3,np.linalg.inv(Kr2))

            # load image
            left_image_dir = seq_path / 'images/left/rectified'
            right_image_dir = seq_path / 'images/right/rectified'
            assert left_image_dir.is_dir()
            assert right_image_dir.is_dir()
            left_image_pathstrings = list()
            right_image_pathstrings = list()

            if self.mode == 'test':
                # for entry in left_image_dir.iterdir():
                #     assert str(entry.name).endswith('.png')
                #     if int((entry.name).split('.')[0]) in self.image_index:
                #         left_image_pathstrings.append(str(entry))
                # left_image_pathstrings.sort()
                # self.left_image_pathstrings = left_image_pathstrings

                # for entry in right_image_dir.iterdir():
                #     assert str(entry.name).endswith('.png')
                #     if int((entry.name).split('.')[0]) in self.image_index:
                #         right_image_pathstrings.append(str(entry))
                # right_image_pathstrings.sort()
                # self.right_image_pathstrings = right_image_pathstrings
                
                for entry in left_image_dir.iterdir():
                    assert str(entry.name).endswith('.png')
                    # if int((entry.name).split('.')[0]) % 2 == 0:
                    left_image_pathstrings.append(str(entry))
                left_image_pathstrings.sort()
                self.left_image_pathstrings = left_image_pathstrings

                for entry in right_image_dir.iterdir():
                    assert str(entry.name).endswith('.png')
                    # if int((entry.name).split('.')[0]) % 2 == 0:
                    right_image_pathstrings.append(str(entry))
                right_image_pathstrings.sort()
                self.right_image_pathstrings = right_image_pathstrings

                assert int(Path(self.left_image_pathstrings[0]).stem) == 0
                assert int(Path(self.right_image_pathstrings[0]).stem) == 0
                self.left_image_pathstrings.pop(0)
                self.right_image_pathstrings.pop(0)
                
            else:
                for entry in left_image_dir.iterdir():
                    assert str(entry.name).endswith('.png')
                    # if int((entry.name).split('.')[0]) % 2 == 0:
                    left_image_pathstrings.append(str(entry))
                left_image_pathstrings.sort()
                self.left_image_pathstrings = left_image_pathstrings

                for entry in right_image_dir.iterdir():
                    assert str(entry.name).endswith('.png')
                    # if int((entry.name).split('.')[0]) % 2 == 0:
                    right_image_pathstrings.append(str(entry))
                right_image_pathstrings.sort()
                self.right_image_pathstrings = right_image_pathstrings

                assert int(Path(self.left_image_pathstrings[0]).stem) == 0
                assert int(Path(self.right_image_pathstrings[0]).stem) == 0
                self.left_image_pathstrings.pop(0)
                self.right_image_pathstrings.pop(0)
                # breakpoint()
                # assert len(self.right_image_pathstrings) //2 == self.timestamps.size
                # assert len(self.left_image_pathstrings) //2 == self.timestamps.size

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        # print(self.disp_gt_pathstrings[0])
        if self.mode != 'test':
            assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
            self.disp_gt_pathstrings.pop(0)
            self.timestamps = self.timestamps[1:]

        # self.h5f = dict()
        # self.rectify_ev_maps = dict()
        # self.event_slicers = dict()
        
        self.voxel_path = dict()
        # voxel_dir = raw_seq_path / 'raw_events'
        
        self.raw_event_dir = raw_seq_path / 'raw_events'
        
        voxel_dir = raw_seq_path / 'voxel_50ms_15bin'
        # load image
        left_voxel_dir = voxel_dir / 'left'
        right_voxel_dir = voxel_dir / 'right'
        
        assert left_voxel_dir.is_dir()
        assert right_voxel_dir.is_dir()
        left_voxel_pathstrings = list()
        right_voxel_pathstrings = list()
        
        for entry in left_voxel_dir.iterdir():
            assert str(entry.name).endswith('.npy')
            left_voxel_pathstrings.append(str(entry))
        left_voxel_pathstrings.sort()
        self.left_voxel_pathstrings = left_voxel_pathstrings

        for entry in right_voxel_dir.iterdir():
            assert str(entry.name).endswith('.npy')
            right_voxel_pathstrings.append(str(entry))
        right_voxel_pathstrings.sort()
        self.right_voxel_pathstrings = right_voxel_pathstrings

        
        assert len(self.right_voxel_pathstrings) //2 == self.timestamps.size
        assert len(self.left_voxel_pathstrings) //2 == self.timestamps.size
        
        # breakpoint()
        # self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # self.subset_root = os.path.join('../..', 'val')
        # self.disp_key = 'disparity_image/'
        # self.gt_disp_key = 'gt_disp/'
        # self.left_image_key = 'image0'

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        
        gt_disp = (np.array(disp_16bit).astype(np.float32))/256
        return torch.tensor(gt_disp)

        # @staticmethod
    def get_left_image(self, filepath: Path):
        assert filepath.is_file()   
        # image = imageio.imread(str(filepath), pilmode='RGB') 
        image = np.array(PIL.Image.open(str(filepath))).astype(np.uint8)
        warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography_left, (1440, 1080),  flags=cv2.WARP_INVERSE_MAP)
        warp_image=warp_image[0:480, 0:640,:]
        return warp_image

    # @staticmethod
    def get_right_image(self, filepath: Path):
        assert filepath.is_file()   
        # image = imageio.imread(str(filepath), pilmode='RGB') 
        image = np.array(PIL.Image.open(str(filepath))).astype(np.uint8)
        warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography_right, (1440, 1080))
        warp_image=warp_image[0:480, 0:640,:]
        return warp_image
    
    # @staticmethod
    def get_voxel(self, filepath: Path):
 
        voxel = np.load(str(filepath)) #['voxel']
        return voxel

    def __len__(self):
        # return len(self.disp_gt_pathstrings)
        if self.mode != 'test':
            return len(self.timestamps) + (self.frame_idxs[0]+ 1)//2 
            # return 4
        else:
            return len(self.timestamps_test)

    def add_padding(self, image, pad=0):
        H, W = image.shape[-2:]
        pad_left = 0
        pad_right = self.pad_width - W
        pad_top = self.pad_height - H
        pad_bottom = 0
        assert pad_left >= 0 and pad_right >= 0 and pad_top >= 0 and pad_bottom >= 0, "Require image crop, please check the image size"
        padded = torch.nn.functional.pad(image, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=pad)
        return padded, (pad_left, pad_right, pad_top, pad_bottom)

    def __getitem__(self, index):
        data = []
        
        # print('-------------')
        # print("############Start##############")
        # breakpoint()
        if self.mode != 'test':
            index = index + len(self.frame_idxs)//2 + len(self.frame_idxs)%2 - 1
            index = 2 * index + 1
        else:
            index = self.image_index[index] - 1 
        
        for idx in self.frame_idxs:
            debug = {}
            # ts_end = self.timestamps[index + idx]
            # ts_start should be fine (within the window as we removed the first disparity map)
            # ts_start = ts_end - self.delta_t_us
            if self.mode != 'test':
                if (idx % 2) == 0:
                    disp_gt_path = Path(self.disp_gt_pathstrings[index//2+idx//2])
                    disparity_gt = self.get_disparity_map(disp_gt_path)
                else:
                    disparity_gt = torch.zeros(self.height, self.width) + float('Nan')
                    
                output = {
                    'disparity_gt': disparity_gt
                }
            else:
                output = {}
                # save_path = self.left_voxel_pathstrings[index+idx].\
                #  split('DSEC_voxel_raw/')[-1].replace('/voxel_50ms_15bin/left', '').replace('npy', 'png')
                save_path = os.path.join(self.left_voxel_pathstrings[index+idx].split('/')[-4], 
                                         self.left_voxel_pathstrings[index+idx].split('/')[-1]).replace('npy', 'png')
            
            
            if self.img_load:  
                # breakpoint()
                # if self.mode != 'test':
                #     i = index//2+idx//2
                # else:
                i = index+idx
                left_image_path = Path(self.left_image_pathstrings[i])
                # print(left_image_path)
                right_image_path = Path(self.right_image_pathstrings[i]) 
                left_image = self.get_left_image(left_image_path)/255.0
                right_image = self.get_right_image(right_image_path)/255.0
                output['left_image'] = common.image2tensor(left_image).type(torch.float32)
                output['right_image'] = common.image2tensor(right_image).type(torch.float32)
            
            
            for location in self.locations:
                if location == 'left':
                    voxel_path = self.left_voxel_pathstrings[index+idx]
                    # print(left_image_path)
                    # print(voxel_path)
                    # print("-------------------------")
                elif location == 'right':
                    voxel_path= self.right_voxel_pathstrings[index+idx]
                
                # raw_events = self.get_voxel()
                event_representation = torch.tensor(self.get_voxel(voxel_path))
                if self.mode != 'test':
                    if (idx % 2) == 0:
                        raw_event_path = voxel_path.replace('voxel_50ms_15bin', 'raw_events')
                        raw_events = self.get_voxel(raw_event_path)
                        t = raw_events[:, 0]
                        x_rect = raw_events[:, 1]
                        y_rect = raw_events[:, 2]
                        p = raw_events[:, 3]
                        
                        event_index = (x_rect < self.width) & (y_rect < self.height)
                        t = t[event_index]
                        x_rect = x_rect[event_index]
                        y_rect = y_rect[event_index]
                        p = p[event_index]
                        
                        contrast_data = self.event_sequence_loader.get_from_data(x_rect, y_rect, t, p, None)
                        if 'contrast' not in output:
                            output['contrast'] = dict()
                        output['contrast'][location] = contrast_data
                    else:
                        pass
                
                if 'representation' not in output:
                    output['representation'] = dict()
                output['representation'][location] = event_representation
            left_padded, pad = self.add_padding(output['representation']['left'])
            right_padded, _ = self.add_padding(output['representation']['right'])
            
            if self.mode != 'test':
                disp_padded , _  = self.add_padding(output['disparity_gt'], 255.0)
                
                
                disp_mask = disp_padded > self.eval_maxdisp
                disp_padded[disp_mask] = float('Nan')
                disp_mask = (disp_padded == 0)
                disp_padded[disp_mask] = float('Nan')
            
            
          
            if self.mode == 'train':
                # pseudo_gt_path = os.path.join(self.pseudo_path, os.path.join(*str(self.seq_path).split('/')[-2:]), str(disp_gt_path).split('event/')[1])
                pseudo_gt_path = os.path.join(self.pseudo_path, os.path.join(*str(self.seq_path).split('/')[-2:]), 
                                              self.left_voxel_pathstrings[index+idx].split('/')[-1].replace('.npy', '.png'))
                
                
                pseudo_gt = self.get_disparity_map(Path(pseudo_gt_path))
                pseudo_gt[:, :45] = float('Nan')
                pseudo_gt[400:, :] = float('Nan')
                pseudo_gt_padded , _  = self.add_padding(pseudo_gt, 255.0)
                
                pseudo_gt_mask = pseudo_gt_padded  > self.eval_maxdisp
                pseudo_gt_padded[pseudo_gt_mask] = float('Nan')
                pseudo_gt_mask = (pseudo_gt_padded == 0)
                pseudo_gt_padded[pseudo_gt_mask] = float('Nan')
                
                debug['pseudo_disp'] = pseudo_gt_padded
                
                if (idx % 2) == 0:
                    debug['left_list'] = output['contrast']['left']['inp_list']
                    debug['left_pol_mask'] = output['contrast']['left']['inp_pol_mask']
                    debug['right_list'] = output['contrast']['right']['inp_list']
                    debug['right_pol_mask'] = output['contrast']['right']['inp_pol_mask']
            
                # if (idx % 2) == 0:
                #     print(str(disp_gt_path).split('/')[-1], str(pseudo_gt_path).split('/')[-1], 
                #           str(voxel_path).split('/')[-1], str(raw_event_path).split('/')[-1])
                # else:
                #     print(str(pseudo_gt_path).split('/')[-1], str(voxel_path).split('/')[-1])
                
            # print(self.mode)
            # if self.mode == 'train':
                
            #     if (idx % 2) == 0:
            #         print(self.left_voxel_pathstrings[index+idx], self.disp_gt_pathstrings[index//2+idx//2])
            #     else:
            #         print(self.left_voxel_pathstrings[index+idx], pseudo_gt_path)
            # else:
            #     if idx == 0:
            #         print(self.left_voxel_pathstrings[index+idx], self.disp_gt_pathstrings[index//2+idx//2])
            #     else:
            #         print(self.left_voxel_pathstrings[index+idx])
            if self.img_load:
                debug['left_img'] = output['left_image']
                debug['right_img'] = output['right_image']
            debug["left_event_path"]= self.left_voxel_pathstrings[index+idx]
            if self.mode != 'test':
                data.append((left_padded, right_padded, disp_padded, pad, debug))
            else:
                data.append((left_padded, right_padded, pad, debug, save_path))
        # return output
        return data
    
