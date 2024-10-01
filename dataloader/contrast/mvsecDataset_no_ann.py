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
from .raw_event_utils import EventSequenceLoader
from torch.utils.data.dataloader import default_collate

FRAMES_FILTER_FOR_TEST = {
    'indoor_flying': {
        1: list(range(140, 1201)),
        2: list(range(120, 1421)),
        3: list(range(73, 1616)),
        # 3: list(range(107, 1750)),
        4: list(range(190, 290))
    }
}
FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying': {
        1: list(range(80, 1260)),
        2: list(range(160, 1580)),
        3: list(range(125, 1815)),
        4: list(range(190, 290))
    }
}
FRAMES_FILTER_FOR_ALL = {
    'indoor_flying': {
        1: list(range(0, 1398)),
        2: list(range(0, 1690)),
        3: list(range(0, 1873))
    }
}

# FRAMES_FILTER_FOR_TEST = {
#     'indoor_flying': {
#         1: list(range(170, 172)),
#         2: list(range(170, 172)),
#         3: list(range(170, 172)),
#         # 3: list(range(107, 1750)),
#         4: list(range(170, 172))
#     }
# }
# FRAMES_FILTER_FOR_TRAINING = {
#     'indoor_flying': {
#         1: list(range(170, 172)),
#         2: list(range(170, 172)),
#         3: list(range(170, 172)),
#         4: list(range(170, 172))
#     }
# }
# FRAMES_FILTER_FOR_ALL = {
#     'indoor_flying': {
#         1: list(range(170, 172)),
#         2: list(range(170, 172)),
#         3: list(range(170, 172))
#     }
# }

class MVSECdataset(Dataset):
    def __init__(self, root: str, ann: list, height: int, width: int, 
                frame_idxs: list = [0, ], eval_maxdisp=255.0, 
                skip=1, load_pseudo_gt=False, load_image = False):
        self.root = root
        self.data_list =  ann
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.eval_maxdisp = eval_maxdisp
        self.skip = skip
        self.load_pseudo_gt = load_pseudo_gt
        self.load_img = load_image

        event_sequence_loader_config={
            'root': root,
            'resolution': [height, width],
        }
        self.event_sequence_loader = EventSequenceLoader(event_sequence_loader_config)
    # def annLoader(self, annFile):
    #     annFile = os.path.join(self.root, annFile)
    #     assert os.path.isfile(annFile)
    #     data_list = []
    #     with open(file=annFile, mode='r') as fp:
    #         data_list.extend(json.load(fp))
    #     return data_list
    
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
        # print('#############################')
        # pdb.set_trace()
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
            pseudo_disp_path = frame_info.get('pseudo_disp_path', None)
            left_raw_path = frame_info.get('left_raw_path', None)
            right_raw_path = frame_info.get('right_raw_path', None)
            
            left_event_path = os.path.join(self.root, left_event_path)
            right_event_path = os.path.join(self.root, right_event_path)
            # print(left_event_path)
            disp_path = os.path.join(self.root, disp_path)
            flow_path = os.path.join(self.root, flow_path)
            pseudo_disp_path = os.path.join(self.root, pseudo_disp_path)
            left_raw_path = os.path.join(self.root, left_raw_path)
            right_raw_path = os.path.join(self.root, right_raw_path)

            left_event = np.load(left_event_path)
            right_event = np.load(right_event_path)
            # need to be (C, H, W)
            assert left_event.ndim==3
            assert right_event.ndim==3
            # flow_gt = np.load(flow_path)
            # pdb.set_trace()
            # assert flow_gt.ndim==3
            
            # (H, W)
            disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
            # wierd behavior of mvsec, just do it
            disp = disp / 7.0

            left_event = torch.tensor(left_event)
            right_event = torch.tensor(right_event)
            disp = torch.tensor(disp)
            # flow_gt = torch.tensor(flow_gt)
                        
            left_padded, pad = self.add_padding(left_event)
            right_padded, _ = self.add_padding(right_event)
            disp_padded , _  = self.add_padding(disp, 255.0)
            # flow_gt, _ = self.add_padding(flow_gt)
                        
            disp_mask = disp_padded > self.eval_maxdisp
            disp_padded[disp_mask] = float('Nan')
            # disp_padded[disp_mask] = float(1e5)          
            
            debug = {"left_event_path": left_event_path,
                "right_event_path": right_event_path, 
                "disp_path": disp_path,
                # "flow_gt": flow_gt,
                }
            
            if idx <= 0:
                # breakpoint()
                left_raw = self.event_sequence_loader.get(left_raw_path, pad)
                right_raw = self.event_sequence_loader.get(right_raw_path, pad)

                ############ TODO: remove later#######
                # left_padded = left_raw['inp_voxel']
                # right_padded = right_raw['inp_voxel']
                #################################
                debug['left_list'] = left_raw['inp_list']
                debug['left_pol_mask'] = left_raw['inp_pol_mask']
                debug['right_list'] = right_raw['inp_list']
                debug['right_pol_mask'] = right_raw['inp_pol_mask']

            if self.load_pseudo_gt:
                pseudo_disp = np.load(pseudo_disp_path)
                pseudo_disp = torch.tensor(pseudo_disp)
                pseudo_disp, _ = self.add_padding(disp, 255.0)
                debug["pseudo_disp"] = pseudo_disp.type(torch.float32)
            if self.load_img:
                left_img_path = left_event_path.replace('voxel0skip' + str(self.skip) + 'bin5', 'image0').replace('npy', 'png')
                right_img_path = right_event_path.replace('voxel1skip' + str(self.skip) + 'bin5', 'image1').replace('npy', 'png')
                left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
                right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
                left_img = torch.tensor(left_img).type(torch.float32)/255.0
                right_img = torch.tensor(right_img).type(torch.float32)/255.0
                left_img_padded, _ = self.add_padding(left_img)
                right_img_padded, _ = self.add_padding(right_img)
                debug["left_img"] = left_img_padded
                debug["right_img"] = right_img_padded
                
            data.append((left_padded, right_padded, disp_padded, pad, debug))
        return data

def get_contrast_train_dataset(root, split, height, width, frame_idxs, eval_maxdisp, skip_num, load_pseudo_gt, load_image = False):
    assert max(frame_idxs) == 0
    # frame_idxs.sort()

    # import pdb; pdb.set_trace()

    assert os.path.isdir(root)
    if split==1 or split=='1':
        train_seqs = [2, 3]
        test_seqs = [1]
        pseudo_disp_dir_name = 'pseudo_disp_split1'
    elif split==3 or split=='3':
        train_seqs = [1, 2]
        test_seqs = [3]
        pseudo_disp_dir_name = 'pseudo_disp_split3'
    elif split=='1,2':
        train_seqs = [1, 2]
        test_seqs = [1, 2]
    elif split=='2,3':
        train_seqs = [2, 3]
        test_seqs = [2, 3]
    elif split=='all' or split=='All':
        train_seqs = [1, 2, 3]
        test_seqs = [1, 2, 3]
        pseudo_disp_dir_name = 'pseudo_disp_split3'
    else:
        raise NotImplementedError
    
    train_test_file_dict = []
    for train_test in ['train', 'test']:
        if train_test == 'train':
            seqs = train_seqs
            frame_filter = FRAMES_FILTER_FOR_TRAINING['indoor_flying']
        elif train_test == 'test':
            seqs = test_seqs
            frame_filter = FRAMES_FILTER_FOR_TEST['indoor_flying']
            # frame_filter = FRAMES_FILTER_FOR_ALL['indoor_flying']
        else:
            raise NotImplementedError
        
        file_dict = []
        # make file sub-path for train
        for seq in seqs:
            seq_dir = f'indoor_flying_{seq}'
            
            left_dir_path = os.path.join(seq_dir, 'voxel0skip' + str(skip_num) + 'bin5')
            right_dir_path = os.path.join(seq_dir, 'voxel1skip' + str(skip_num) + 'bin5')
        
            disp_dir_path = os.path.join(seq_dir, 'disparity_image')
            flow_dir_path = os.path.join(seq_dir, 'flow0')

            left_rawevent_dir_path = os.path.join(seq_dir, 'event0')
            right_rawevent_dir_path = os.path.join(seq_dir, 'event1')

            pseudo_disp_dir_path = os.path.join(seq_dir, pseudo_disp_dir_name)

            for frame in frame_filter[seq][-min(frame_idxs):]:
                left_right_disp = {}
                for fi in frame_idxs:
                    num_str = "{:06d}".format(frame + fi)
                    event_name = num_str + ".npy"
                    disp_name = num_str + ".png"
                    
                    if not os.path.isfile(os.path.join(root, left_dir_path, event_name)):
                        print(os.path.join(root, left_dir_path, event_name))
                    assert os.path.isfile(os.path.join(root, right_dir_path, event_name))
                    
                    assert os.path.isfile(os.path.join(root, disp_dir_path, disp_name))
                    left_right_disp[str(fi)] = {
                        'left_image_path': os.path.join(left_dir_path, event_name),
                        'right_image_path': os.path.join(right_dir_path, event_name),
                        'left_disp_path': os.path.join(disp_dir_path, disp_name),
                        'left_flow_path': os.path.join(flow_dir_path, event_name),
                        'pseudo_disp_path': os.path.join(pseudo_disp_dir_path, event_name),
                        'left_raw_path': os.path.join(left_rawevent_dir_path, event_name),
                        'right_raw_path': os.path.join(right_rawevent_dir_path, event_name)
                    }     
                file_dict.append(left_right_disp)
        train_test_file_dict.append(file_dict)
    
    train_dataset = MVSECdataset(
        root,
        train_test_file_dict[0],
        height,
        width,
        frame_idxs,
        eval_maxdisp,
        skip_num,
        load_pseudo_gt,
        load_image = load_image)
    
    return train_dataset
# B x frame x (datas) -> frame x B*(datas)
def custom_collate(batch): # batch = ((left_padded, right_padded, disp_padded, pad, debug, "left_inp_list", "left_inp_pol_mask", "right_inp_list", "right_inp_pol_mask))
    """
    Collects the different event representations and stores them together in a dictionary.
    """
    B = len(batch)
    num_frame = len(batch[0])
    
    datasets = [([[] for i in range(4)] + [{}]) for i in range(num_frame)]

    for bn, data in enumerate(batch):
        for f, d in enumerate(data):
            datasets[f][0].append(d[0])
            datasets[f][1].append(d[1])
            datasets[f][2].append(d[2])
            datasets[f][3].append(d[3])
            if bn == 0:
                for key in d[4].keys():
                    datasets[f][4][key] = []
            for key in d[4].keys():
                datasets[f][4][key].append(d[4][key])
    
    for fi, frame_data in enumerate(datasets):
        for i, data in enumerate(frame_data):
            if isinstance(data, dict):
                for key in data.keys():
                    try:
                        frame_data[i][key] = default_collate(data[key])
                    except:
                        continue
            else: 
                frame_data[i] = default_collate(data)
    
    return datasets

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
