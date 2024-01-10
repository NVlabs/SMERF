import cv2
import os
import argparse
import pickle
import sparse

import os.path as osp
import numpy as np

from openlanev2.centerline.dataset import Collection
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

BEV_SCALE = 4
BEV_RANGE = [-100, 100, -50, 50]
THICKNESS = 1

def raster_sd_map(sd_map, line_weight=9):
    if sd_map is not None:
        raster = []
        for i, category in enumerate(sd_map):
            image = np.ones((
                BEV_SCALE * (BEV_RANGE[1] - BEV_RANGE[0]),
                BEV_SCALE * (BEV_RANGE[3] - BEV_RANGE[2]),
            ), dtype=np.int32) * 0
            for road in sd_map[category]:
                road = (BEV_SCALE * (-road[:, :2] + np.array([BEV_RANGE[1] , BEV_RANGE[3]]))).astype(int)
                cv2.polylines(image, [road[:, [1,0]]], False, 1, THICKNESS * line_weight)
            raster.append(image)
        raster = np.stack(raster, axis=-1)
    else:
        raster = np.zeros((
            BEV_SCALE * (BEV_RANGE[1] - BEV_RANGE[0]),
            BEV_SCALE * (BEV_RANGE[3] - BEV_RANGE[2]),
            8,
        ), dtype=np.int32)
    return raster


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rasterized SD maps.')
    parser.add_argument('--collection', type=str, default='data_dict_sample_train', required=False,
                        help='split to process raster SD maps.')
    parser.add_argument('--map_dir_prefix', type=str, default='sd_map_reraster', required=False)
    parser.add_argument('--filter', type=str, default=None, required=False)
    parser.add_argument('--total_parts', type=int, default=1, required=False)
    parser.add_argument('--part', type=int, default=0, required=False)

    args = parser.parse_args()

    root_path = '../data/OpenLane-V2'
    collection = Collection(root_path, root_path, args.collection)
    line_weight = 20 if args.filter == 'idt' else 10 

    if args.total_parts > 1:
        num_frames_ppart = len(collection.frames) // args.total_parts + 1
        raster_frames = list(collection.frames.items())[num_frames_ppart * args.part: num_frames_ppart * (args.part + 1)]
    else:
        raster_frames = collection.frames.items()

    for frame_key, frame in tqdm(raster_frames):
        map_crop_dir = osp.join(root_path, args.map_dir_prefix, frame_key[0], frame_key[1])
        os.makedirs(map_crop_dir, exist_ok=True)

        map_crop_path = osp.join(map_crop_dir, f"{frame_key[-1]}.npz")
        if osp.exists(map_crop_path):
            continue

        graph = pickle.load(open(f'../data/OpenLane-V2/sd_map_graph_all/{frame_key[0]}/{frame_key[1]}/{frame_key[2]}.pkl', "rb"))
        
        map_raster = raster_sd_map(graph, line_weight=line_weight)
        
        if args.filter is not None:
            map_raster_filter = []

            # gaussian
            if args.filter == 'gaussian':
                for i in range(map_raster.shape[-1]):
                    map_blur_sd = gaussian_filter(map_raster[..., i].astype(np.float32), sigma=3)
                    map_raster_filter.append(map_blur_sd)

            # distance transform
            elif args.filter == 'idt':
                for i in range(map_raster.shape[-1]):
                    im = np.array(map_raster[..., i] * 255, dtype = np.uint8)

                    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

                    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
                    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
                    map_raster_filter.append(dist)

            map_raster = np.stack(map_raster_filter, axis=-1).astype(np.float32)

        # save image
        sparse_map_raster = sparse.COO(map_raster)
        sparse.save_npz(map_crop_path, sparse_map_raster)
    
    print("Finished parsing, saved to", map_crop_dir)
