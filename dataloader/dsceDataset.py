import torch
import torch.nn.functional
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os  
import json
import cv2
import random
import pdb


class DSECdataset(Dataset):
    def __init__(self, root: str, ann: list, height: int, width: int, frame_idxs: list = [0, ], eval_maxdisp=255.0):
        self.root = root
        self.data_list =  ann
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.eval_maxdisp = eval_maxdisp

    
    def add_padding(self, image, pad=0):
        H, W = image.shape[-2:]
        pad_left = 0
        pad_right = self.width - W
        pad_top = self.height - H
        pad_bottom = 0
        assert pad_left >= 0 and pad_right >= 0 and pad_top >= 0 and pad_bottom >= 0, "Require image crop, please check the image size"
        padded = torch.nn.functional.pad(image, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=pad)
        return padded, (pad_left, pad_right, pad_top, pad_bottom)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = []
        # get dictionary of frame idxs
        metadata = self.data_list[i]
        for idx in self.frame_idxs:
            frame_info = metadata.get(str(idx), None)
            # get relative file path respect to self.root
            assert frame_info is not None, "Cannot get frame info of frame {:d}:{:d}, check json file".format(i, idx)
            left_event_path = frame_info.get('left_image_path', None)
            assert left_event_path is not None
            right_event_path = frame_info.get('right_image_path', None)
            assert right_event_path is not None
            disp_path = frame_info.get('left_disp_path', None)
            assert disp_path is not None
            flow_path = frame_info.get('left_flow_path', None)
            assert flow_path is not None
            
            left_event_path = os.path.join(self.root, left_event_path)
            right_event_path = os.path.join(self.root, right_event_path)
            disp_path = os.path.join(self.root, disp_path)
            flow_path = os.path.join(self.root, flow_path)
            left_img_path = left_event_path.replace('num5voxel0', 'image0').replace('npy', 'png')
            right_img_path = right_event_path.replace('num5voxel1', 'image1').replace('npy', 'png')
            
            left_event = np.load(left_event_path)
            right_event = np.load(right_event_path)
            # need to be (C, H, W)
            assert left_event.ndim==3
            assert right_event.ndim==3
            flow_gt = np.load(flow_path)
            # pdb.set_trace()
            assert flow_gt.ndim==3
            
            # (H, W)
            disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
            # wierd behavior of mvsec, just do it
            disp = disp / 7.0
            
            left_event = torch.tensor(left_event)
            right_event = torch.tensor(right_event)
            disp = torch.tensor(disp)
            flow_gt = torch.tensor(flow_gt)
            left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
            left_img = torch.tensor(left_img).type(torch.float32)/255.0
            right_img = torch.tensor(right_img).type(torch.float32)/255.0
            
            left_padded, pad = self.add_padding(left_event)
            right_padded, _ = self.add_padding(right_event)
            disp_padded , _  = self.add_padding(disp, 255.0)
            flow_gt, _ = self.add_padding(flow_gt)
            left_img_padded, _ = self.add_padding(left_img)
            right_img_padded, _ = self.add_padding(right_img)
            
            disp_mask = disp_padded > self.eval_maxdisp
            disp_padded[disp_mask] = float('Nan')
            # disp_padded[disp_mask] = float(1e5)          
            
            debug = {"left_event_path": left_event_path,
                "right_event_path": right_event_path, 
                "disp_path": disp_path,
                "flow_gt": flow_gt,
                "left_img": left_img,
                "right_img": right_img}
            data.append((left_padded, right_padded, disp_padded, pad, debug))
        return data

def get_datasets(root, split, height, width, frame_idxs, num_validation, eval_maxdisp):
    assert max(frame_idxs) == 0
    # frame_idxs.sort()
    assert os.path.isdir(root)
   
    
    train_test_file_dict = []
    for train_test in ['train', 'test']:
        if train_test == 'train':
            seqs = train_seqs
            frame_filter = FRAMES_FILTER_FOR_TRAINING['indoor_flying']
        elif train_test == 'test':
            seqs = test_seqs
            frame_filter = FRAMES_FILTER_FOR_TEST['indoor_flying']
        else:
            raise NotImplementedError
        
        file_dict = []
        # make file sub-path for train
        for seq in seqs:
            seq_dir = f'indoor_flying_{seq}'
            left_dir_path = os.path.join(seq_dir, 'num5voxel0')
            right_dir_path = os.path.join(seq_dir, 'num5voxel1')
            disp_dir_path = os.path.join(seq_dir, 'disparity_image')
            flow_dir_path = os.path.join(seq_dir, 'flow0')
            for frame in frame_filter[seq][-min(frame_idxs):]:
                left_right_disp = {}
                for fi in frame_idxs:
                    num_str = "{:06d}".format(frame + fi)
                    event_name = num_str + ".npy"
                    disp_name = num_str + ".png"
                    assert os.path.isfile(os.path.join(root, left_dir_path, event_name))
                    assert os.path.isfile(os.path.join(root, right_dir_path, event_name))
                    assert os.path.isfile(os.path.join(root, disp_dir_path, disp_name))
                    left_right_disp[str(fi)] = {
                        'left_image_path': os.path.join(left_dir_path, event_name),
                        'right_image_path': os.path.join(right_dir_path, event_name),
                        'left_disp_path': os.path.join(disp_dir_path, disp_name),
                        'left_flow_path': os.path.join(flow_dir_path, event_name)
                    }     
                file_dict.append(left_right_disp)
        train_test_file_dict.append(file_dict)
    
    train_dataset = MVSECdataset(
        root,
        train_test_file_dict[0],
        height,
        width,
        frame_idxs,
        eval_maxdisp)
    
    if num_validation > 0:
        import random
        random.shuffle(train_test_file_dict[1])
        validation_dataset = MVSECdataset(
            root,
            train_test_file_dict[1][:200],
            height,
            width,
            frame_idxs,
            eval_maxdisp)
        test_dataset = MVSECdataset(
            root,
            train_test_file_dict[1][200:],
            height,
            width,
            frame_idxs,
            eval_maxdisp)
    else:
        validation_dataset = DSECdataset(
            root,
            train_test_file_dict[1],
            height,
            width,
            frame_idxs,
            eval_maxdisp)
        test_dataset = DSECdataset(
            root,
            train_test_file_dict[1],
            height,
            width,
            frame_idxs,
            eval_maxdisp)
    
    return train_dataset, validation_dataset, test_dataset


# for debugging
if __name__ == '__main__':
    # mvsecdataset = MVSECdataset(
    #     '/home/jaeyoung/data/ws/event_stereo_ICCV2019/dataset',
    #     '/home/jaeyoung/data/ws/event_stereo_ICCV2019/dataset/view_4_train_v5_split1.json',
    #     288,
    #     352
    # )
    # test = mvsecdataset[0]
    # breakpoint()
    new_train, new_val, new_test = get_datasets('/home/jaeyoung/data/ws/event_stereo_ICCV2019/dataset', 1, 288, 352, [-3, -2, -1, 0], 0)
    # min_disp = []
    # for i in new_train:
    #     min_disp.append(torch.min(i[-1][2]).item())
    new_train[0]
    breakpoint()
    # getDataset(root, 'event_left', 'event_right', 'disparity_image', 80, 1260)
