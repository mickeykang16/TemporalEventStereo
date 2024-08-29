import os
import numpy as np
import pdb
import torch
from pathlib import Path
import hdf5plugin
import h5py
from eventslicer import EventSlicer
import argparse

num_bins = 15
height = 480
width = 640


def events_to_voxel_grid(x, y, p, t, num_bins=num_bins, width=width, height=height):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    pol = torch.from_numpy(pol)
    time = torch.from_numpy(t)
    
    with torch.no_grad():
        voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float, requires_grad=False)
        C, H, W = voxel_grid.shape
        t_norm = time
        t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()
        if int(pol.min()) == -1: 
            value = pol
        else:
            value = 2*pol-1
        # import pdb; pdb.set_trace()
        for xlim in [x0,x0+1]:
            for ylim in [y0,y0+1]:
                for tlim in [t0,t0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
                    interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())
                    index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean
    
    return voxel_grid

def rectify_events(x: np.ndarray, y: np.ndarray, location: str, rectify_ev_maps):
    rectify_map = rectify_ev_maps[location]
    assert rectify_map.shape == (height, width, 2), rectify_map.shape
    assert x.max() < width
    assert y.max() < height
    return rectify_map[y, x]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Dataset Processing.')
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    args = parser.parse_args()
    
    dataset_dir = args.dataset_path
    save_dir = dataset_dir
    
    folder_list_all = os.listdir(dataset_dir)
    folder_list_all.sort()
    
    event_prefix = 'events'
    
    locations = ['left', 'right']
    delta_t_ms = 50
    delta_t_us = delta_t_ms * 1000
    
    for folder_name in folder_list_all:
        print(folder_name)
        
        seq_path = Path(os.path.join(dataset_dir, folder_name))
       
        
        img_dir = seq_path / 'images'
        assert img_dir.is_dir()
        
        timestamps = np.loadtxt(img_dir / 'timestamps.txt', dtype='int64')


        img_left_dir = img_dir / 'left/rectified'
        img_pathstrings = list()
        for entry in img_left_dir.iterdir():
            assert str(entry.name).endswith('.png')
            img_pathstrings.append(str(entry))
        img_pathstrings.sort()


        # timestamps = timestamps[::2]
        timestamps = timestamps[1:]
        img_pathstrings = img_pathstrings[1:]
        assert len(img_pathstrings) == timestamps.size
        
        event_dir = seq_path / event_prefix
        assert event_dir.is_dir()
                
        h5f = dict()
        rectify_ev_maps = dict()
        event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            h5f[location] = h5f_location
            event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                rectify_ev_maps[location] = h5_rect['rectify_map'][()]
        
        
        event_vox_save_dir = os.path.join(save_dir, folder_name, 'voxel_50ms_15bin')
        if not os.path.exists(event_vox_save_dir):
            os.makedirs(event_vox_save_dir)
        if not os.path.exists(os.path.join(event_vox_save_dir, 'left')):
            os.makedirs(os.path.join(event_vox_save_dir, 'left'))
        if not os.path.exists(os.path.join(event_vox_save_dir, 'right')):
            os.makedirs(os.path.join(event_vox_save_dir, 'right'))
        
        
        for index in range(len(timestamps)):
            # print(str(2 * index + 2).zfill(6))
            ts_end = timestamps[index]
            # ts_start should be fine (within the window as we removed the first disparity map)
            ts_start = ts_end - delta_t_us

            print(ts_end, event_vox_save_dir + '/' + str(location) + '/' + 
                  img_pathstrings[index].split('/')[-1].replace('.png', '.npy'))
            

            for location in locations:
                event_data = event_slicers[location].get_events(ts_start, ts_end)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = rectify_events(x, y, location, rectify_ev_maps)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = events_to_voxel_grid(x_rect, y_rect, p, t)     
                np.save(event_vox_save_dir + '/' + str(location) + '/' + img_pathstrings[index].split('/')[-1].replace('.png', '.npy')
                                    , event_representation)
            
