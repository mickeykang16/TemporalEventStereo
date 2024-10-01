import os
import h5py
import numpy as np

import torch
import torch.utils.data as data

# from .utils import ProgressBar

# from .encodings import binary_search_array
from .encodings import events_to_voxel, events_to_channels, events_to_mask, get_hot_event_mask
from torch.utils.data.dataloader import default_collate


class EventSequenceLoader():
    def __init__(self, config):
    
        self.root_path = config.get("root", '')
        self.resolution = config['resolution']
        self.num_bins = 5

    def get_events(self, path):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """
        file = np.load(path)
        xs = file[:, 1]
        ys = file[:, 2]
        ts = file[:, 0]
        ps = file[:, 3]
        ts -= file[0][0]  # sequence starting at t0 = 0
        ts *= 1.0e6  # us
        return xs, ys, ts, ps

    @staticmethod
    def event_formatting(xs, ys, ts, ps):
        """
        Reset sequence-specific variables.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32))
        if int(ps.min()) == 0:
            ps = 2*ps-1
        ts = (ts - ts[0]) / (ts[-1] - ts[0])
        return xs, ys, ts, ps
    

    def create_cnt_encoding(self, xs, ys, ts, ps):
        """
        Creates a per-pixel and per-polarity event count representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [2 x H x W] event representation
        """

        return events_to_channels(xs, ys, ps, sensor_size=self.resolution)

    def create_voxel_encoding(self, xs, ys, ts, ps):
        """
        Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
        as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
        Depth, and Egomotion', Zhu et al., CVPR'19..
        Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
        Positive events are added as +1, while negative as -1.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [B x H x W] event representation
        """

        return events_to_voxel(
            xs,
            ys,
            ts,
            ps,
            self.num_bins,
            sensor_size=self.resolution,
        )

    @staticmethod
    def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([ts, ys, xs, ps])

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """

        inp_pol_mask = torch.stack([ps, ps])
        inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
        inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
        inp_pol_mask[1, :] *= -1
        return inp_pol_mask
    
    @staticmethod
    def event_padding(xs, ys, ts, ps, pad):
        (pad_left, pad_right, pad_top, pad_bottom) = pad
        return xs+pad_left, ys+pad_top, ts, ps
    
    def get_from_data(self, xs, ys, ts, ps, pad):
      
        # xs, ys, ts, ps = self.get_events(path)

        # timestamp normalization
        xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)
        # padding
        # xs, ys, ts, ps = self.event_padding(xs, ys, ts, ps, pad)

        # data augmentation
        # xs, ys, ps = self.augment_events(xs, ys, ps, batch)

        # artificial pauses to the event stream
        # if "Pause" in self.config["loader"]["augment"]:
        #     if self.batch_augmentation["Pause"]:
        #         xs = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ys = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ts = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ps = torch.from_numpy(np.empty([0]).astype(np.float32))

        # events to tensors
        # inp_cnt = self.create_cnt_encoding(xs, ys, ts, ps)
        # inp_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
        inp_list = self.create_list_encoding(xs, ys, ts, ps)
        inp_pol_mask = self.create_polarity_mask(ps)

        # hot pixel removal
        # if self.config["hot_filter"]["enabled"]:
        #     hot_mask = self.create_hot_mask(xs, ys, ps, batch)
        #     hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
        #     hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        #     inp_voxel = inp_voxel * hot_mask_voxel
        #     inp_cnt = inp_cnt * hot_mask_cnt

    
        # prepare output
        output = {}
        # output["inp_cnt"] = inp_cnt
        # output["inp_voxel"] = inp_voxel

        # 4 x N to N x 4
        output["inp_list"] = inp_list.permute(1, 0)
        output["inp_pol_mask"] = inp_pol_mask.permute(1, 0)

        return output
    
    
    def get_from_path(self, path, pad):

        # load events
        xs = np.zeros((0))
        ys = np.zeros((0))
        ts = np.zeros((0))
        ps = np.zeros((0))

        xs, ys, ts, ps = self.get_events(path)

        # timestamp normalization
        xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)
        # padding
        # xs, ys, ts, ps = self.event_padding(xs, ys, ts, ps, pad)

        # data augmentation
        # xs, ys, ps = self.augment_events(xs, ys, ps, batch)

        # artificial pauses to the event stream
        # if "Pause" in self.config["loader"]["augment"]:
        #     if self.batch_augmentation["Pause"]:
        #         xs = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ys = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ts = torch.from_numpy(np.empty([0]).astype(np.float32))
        #         ps = torch.from_numpy(np.empty([0]).astype(np.float32))

        # events to tensors
        # inp_cnt = self.create_cnt_encoding(xs, ys, ts, ps)
        # inp_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
        inp_list = self.create_list_encoding(xs, ys, ts, ps)
        inp_pol_mask = self.create_polarity_mask(ps)

        # hot pixel removal
        # if self.config["hot_filter"]["enabled"]:
        #     hot_mask = self.create_hot_mask(xs, ys, ps, batch)
        #     hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
        #     hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        #     inp_voxel = inp_voxel * hot_mask_voxel
        #     inp_cnt = inp_cnt * hot_mask_cnt

    
        # prepare output
        output = {}
        # output["inp_cnt"] = inp_cnt
        # output["inp_voxel"] = inp_voxel

        # 4 x N to N x 4
        output["inp_list"] = inp_list.permute(1, 0)
        output["inp_pol_mask"] = inp_pol_mask.permute(1, 0)

        return output

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
