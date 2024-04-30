import os
import os.path as osp
import sys
import torch
import json
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm

try:
    from .dataset_template import DatasetTemplate
except:
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from core.datasets.dataset_template import DatasetTemplate

class S3DIS(DatasetTemplate):
    def __init__(self, cfg, class_choice=None, split='train', load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False):
        super().__init__(cfg = cfg, class_choice=None, split=split, load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False)

        data_root='./data/stanford_indoor3d'
        num_point=2048
        test_area=5
        block_size=1.0
        sample_rate=1.0
        transform=None

        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)
        
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __len__(self):
        return len(self.room_idxs)

    def __getitem__(self, index):
        room_idx = self.room_idxs[index]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        data_dic = {
            'points'    : torch.from_numpy(current_points[:,0:3]).float().cuda(),
            'seg_id'    : torch.from_numpy(current_labels).float().cuda(),
            'norms'     : torch.from_numpy(current_points[:,3:6]).float().cuda() # rgb instead of norms
        }
        
        return data_dic
    
if __name__ == '__main__':
    cfg = dict(
        ROOT_DIR='./',
        DATASET=dict(
            NAME='S3DIS',
            NUM_CLASS=13,
            NUM_POINTS=2048,
            IS_SEGMENTATION=True,
            USE_AUG_JIT=False,
            USE_AUG_ROT=False,
            USE_AUG_TRANS=False,
            USE_RANDOM_SHUFFLE=True,
        )
    )
    from easydict import EasyDict
    cfg = EasyDict(cfg)
    s3dis = S3DIS(cfg)
    