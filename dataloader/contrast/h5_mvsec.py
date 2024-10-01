import os
import h5py
import numpy as np

import torch
import torch.utils.data as data

from .base import BaseDataLoader
from .utils import ProgressBar

from .encodings import binary_search_array

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

class Frames:
    """
    Utility class for reading the APS frames encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]

    def get_frames(self, file, t0, t1, crop, res):
        """
        Get all the APS frames in between two timestamps.
        :param file: file to read from
        :param t0: start time
        :param t1: end time
        :param crop: top-left corner of the patch to be cropped
        :param res: resolution of the patch to be cropped
        :return imgs: list of [H x W] APS images
        :return idx0: index of the first frame
        :return idx1: index of the last frame
        """

        idx0 = binary_search_array(self.ts, t0)
        idx1 = binary_search_array(self.ts, t1)

        imgs = []
        for i in range(idx0, idx1):
            imgs.append(file["images"]["image{:09d}".format(i)][crop[0] : crop[0] + res[0], crop[1] : crop[1] + res[1]])

        return imgs, idx0, idx1


class H5Loader(BaseDataLoader):
    def __init__(self, config, num_bins, list_npy_path):
        super().__init__(config, num_bins)
        self.last_proc_timestamp = 0
        self.list_npy_path = list_npy_path
        self.root_path = config["data"]["path"]
        # "memory" that goes from forward pass to the next
        self.batch_idx = [i for i in range(self.config["loader"]["batch_size"])]  # event sequence
        self.batch_row = [0 for i in range(self.config["loader"]["batch_size"])]  # event_idx / time_idx / frame_idx
        self.batch_t0 = [None for i in range(self.config["loader"]["batch_size"])]
        # breakpoint()
        # input event sequences
        self.files = []
        for root, dirs, files in os.walk(config["data"]["path"]):
            for file in files:
                if file.endswith(".h5") or file.endswith(".hdf5"):
                    self.files.append(os.path.join(root, file))

        # open first files
        # self.open_files = []
        # for batch in range(self.config["loader"]["batch_size"]):
        #     self.open_files.append(h5py.File(self.files[batch], "r"))

        # # load frames from open files
        # self.open_files_frames = []
        # if self.config["data"]["mode"] == "frames":
        #     for batch in range(self.config["loader"]["batch_size"]):
        #         frames = Frames()
        #         self.open_files[batch]["images"].visititems(frames)
        #         self.open_files_frames.append(frames)

        # # progress bars
        # if self.config["vis"]["bars"]:
        #     self.open_files_bar = []
        #     for batch in range(self.config["loader"]["batch_size"]):
        #         max_iters = self.get_iters(batch)
        #         self.open_files_bar.append(ProgressBar(self.files[batch].split("/")[-1], max=max_iters))

    def get_iters(self, batch):
        """
        Compute the number of forward passes given a sequence and an input mode and window.
        """

        if self.config["data"]["mode"] == "events":
            max_iters = len(self.open_files[batch]["events/xs"])
        elif self.config["data"]["mode"] == "time":
            max_iters = self.open_files[batch].attrs["duration"]
        elif self.config["data"]["mode"] == "frames":
            max_iters = len(self.open_files_frames[batch].ts) - 1
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return max_iters // self.config["data"]["window"]

    def get_events(self, index):
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
        file = np.load(os.path.join(self.root_path, self.list_npy_path[index]))
        xs = file[:, 1]
        ys = file[:, 2]
        ts = file[:, 0]
        ps = file[:, 3]
        ts -= file[0][0]  # sequence starting at t0 = 0
        ts *= 1.0e6  # us
        return xs, ys, ts, ps

    def get_event_index(self, batch, window=0):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx = None
        if self.config["data"]["mode"] == "events":
            event_idx = self.batch_row[batch] + window
        elif self.config["data"]["mode"] == "time":
            event_idx = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch]['davis']['left']['events'][0][2] + window
            )
        elif self.config["data"]["mode"] == "frames":
            event_idx = self.find_ts_index(
                self.open_files[batch], self.open_files_frames[batch].ts[self.batch_row[batch] + window]
            )
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        return event_idx

    def find_ts_index(self, file, timestamp):
        """
        Find closest event index for a given timestamp through binary search.
        """
        # breakpoint()
        return binary_search_array(file['davis']['left']['events'][:][2], timestamp)

    def __len__(self):
        return len(self.list_npy_path)
    
    def __getitem__(self, index):
        batch=0
        while True:
            # batch = index % self.config["loader"]["batch_size"]

            # trigger sequence change

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))
                # idx0 = self.get_event_index(batch)
                # idx1 = self.get_event_index(batch, window=self.config["data"]["window"])
            
            # breakpoint()
            xs, ys, ts, ps = self.get_events(index)


            # timestamp normalization

            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # data augmentation
            # xs, ys, ps = self.augment_events(xs, ys, ps, batch)

            # artificial pauses to the event stream
            if "Pause" in self.config["loader"]["augment"]:
                if self.batch_augmentation["Pause"]:
                    xs = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ys = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ts = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ps = torch.from_numpy(np.empty([0]).astype(np.float32))

            # events to tensors
            inp_cnt = self.create_cnt_encoding(xs, ys, ts, ps)
            inp_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            inp_list = self.create_list_encoding(xs, ys, ts, ps)
            inp_pol_mask = self.create_polarity_mask(ps)

            # hot pixel removal
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(xs, ys, ps, batch)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                inp_voxel = inp_voxel * hot_mask_voxel
                inp_cnt = inp_cnt * hot_mask_cnt

            # load frames when required
            if self.config["data"]["mode"] == "frames":
                inp_frames = np.zeros(
                    (
                        2,
                        self.config["loader"]["resolution"][0],
                        self.config["loader"]["resolution"][1],
                    )
                )
                img0 = self.open_files[batch]["images"][self.open_files_frames[batch].names[self.batch_row[batch]]][:]
                img1 = self.open_files[batch]["images"][
                    self.open_files_frames[batch].names[self.batch_row[batch] + self.config["data"]["window"]]
                ][:]
                inp_frames[0, :, :] = self.augment_frames(img0, batch)
                inp_frames[1, :, :] = self.augment_frames(img1, batch)
                inp_frames = torch.from_numpy(inp_frames.astype(np.uint8))

            # update window if not in pause mode
            if "Pause" in self.config["loader"]["augment"]:
                if not self.batch_augmentation["Pause"]:
                    self.batch_row[batch] += self.config["data"]["window"]
            else:
                self.batch_row[batch] += self.config["data"]["window"]

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["inp_cnt"] = inp_cnt
        output["inp_voxel"] = inp_voxel
        output["inp_list"] = inp_list
        output["inp_pol_mask"] = inp_pol_mask
        if self.config["data"]["mode"] == "frames":
            output["inp_frames"] = inp_frames

        return output

def get_datasets(root, split, frame_idxs=[0]):
    
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
            
            left_dir_path = os.path.join(seq_dir, 'event0')
            # right_dir_path = os.path.join(seq_dir, 'voxel1skip' + str(skip_num) + 'bin5')
        
            # disp_dir_path = os.path.join(seq_dir, 'disparity_image')
            # flow_dir_path = os.path.join(seq_dir, 'flow0')

            # pseudo_disp_dir_path = os.path.join(seq_dir, pseudo_disp_dir_name)

            for frame in frame_filter[seq][-min(frame_idxs):]:
                left_right_disp = {}
                for fi in frame_idxs:
                    num_str = "{:06d}".format(frame + fi)
                    event_name = num_str + ".npy"
                    disp_name = num_str + ".png"
                    
                    if not os.path.isfile(os.path.join(root, left_dir_path, event_name)):
                        print(os.path.join(root, left_dir_path, event_name))
                    # assert os.path.isfile(os.path.join(root, right_dir_path, event_name))
                    
                    # assert os.path.isfile(os.path.join(root, disp_dir_path, disp_name))
                    left_right_disp[str(fi)] = {
                        'left_image_path': os.path.join(left_dir_path, event_name),
                        # 'right_image_path': os.path.join(right_dir_path, event_name),
                        # 'left_disp_path': os.path.join(disp_dir_path, disp_name),
                        # 'left_flow_path': os.path.join(flow_dir_path, event_name),
                        # 'pseudo_disp_path': os.path.join(pseudo_disp_dir_path, event_name),
                    }     
                file_dict.append(os.path.join(left_dir_path, event_name))
        train_test_file_dict.append(file_dict)
    return train_test_file_dict