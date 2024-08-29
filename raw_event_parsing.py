import os
import numpy as np
import pdb
import torch
from pathlib import Path
import hdf5plugin
import h5py
from eventslicer import EventSlicer
import torch
import argparse

height = 480
width = 640

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
    
    folder_list_all = os.listdir(dataset_dir)
    folder_list_all.sort()
    
    event_prefix = 'events'
    
    locations = ['left', 'right']
    delta_t_ms = 50

    delta_t_us = delta_t_ms * 1000

    save_dir = dataset_dir
    
    for folder_name in folder_list_all:
        print(folder_name)
        seq_path = Path(os.path.join(dataset_dir, folder_name))
        
        disp_dir = seq_path / 'disparity'
        assert disp_dir.is_dir()
        
        timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load disparity paths
        ev_disp_dir = disp_dir / 'event'
        assert ev_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in ev_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        disp_gt_pathstrings = disp_gt_pathstrings
        assert len(disp_gt_pathstrings) == timestamps.size
        
        event_dir = seq_path / event_prefix
        assert event_dir.is_dir()
        
        
        assert int(Path(disp_gt_pathstrings[0]).stem) == 0
        disp_gt_pathstrings.pop(0)
        timestamps = timestamps[1:]
        
        h5f = dict()
        rectify_ev_maps = dict()
        event_slicers = dict()

        ev_dir = seq_path / 'events'

        event_vox_save_dir = os.path.join(save_dir, folder_name, 'raw_events')
        if not os.path.exists(event_vox_save_dir):
            os.makedirs(event_vox_save_dir)
        else:
            continue
        if not os.path.exists(os.path.join(event_vox_save_dir, 'left')):
            os.makedirs(os.path.join(event_vox_save_dir, 'left'))
        if not os.path.exists(os.path.join(event_vox_save_dir, 'right')):
            os.makedirs(os.path.join(event_vox_save_dir, 'right'))


        for location in locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            h5f[location] = h5f_location
            event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                rectify_ev_maps[location] = h5_rect['rectify_map'][()]
        
        
        
        
        
        
        for index in range(len(timestamps)):
            ts_end = timestamps[index]
            # ts_start should be fine (within the window as we removed the first disparity map)
            ts_start = ts_end - delta_t_us

            for location in locations:
                event_data = event_slicers[location].get_events(ts_start, ts_end)

                p = event_data['p']
                t = event_data['t']

                t = (t - t[0]).astype('uint32')

                x = event_data['x']
                y = event_data['y']

                xy_rect = rectify_events(x, y, location, rectify_ev_maps)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_index = (x_rect >= 0) & (x_rect < width) &  (y_rect >= 0) & (y_rect < height)

                x_rect = x_rect[event_index]
                y_rect = y_rect[event_index]
                t = t[event_index]
                p = p[event_index]


                event_representation = np.stack([t, np.round(x_rect).astype('int'), np.round(y_rect).astype('int'), p], 1).astype('uint32')


                np.save(event_vox_save_dir + '/' + str(location) + '/' + disp_gt_pathstrings[index].split('/')[-1].replace('.png', '.npy'), event_representation)

                
