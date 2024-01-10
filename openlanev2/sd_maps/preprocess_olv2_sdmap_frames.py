import cv2
import os
import argparse
import sparse

import os.path as osp
import numpy as np

# from openlanev2.io import io
from openlanev2.centerline.dataset import Collection
from openlanev2.centerline.visualization import draw_sd_map, rasterize_sd_map
from tqdm import tqdm


def vec_onehot(arr, dim=3):
    one_hot = np.zeros((arr.shape[0], dim), dtype=np.int8)
    one_hot[np.arange(arr.shape[0]), arr.astype(int)] = 1
    return one_hot


def load_sdmap_image(frame, dsize=(400, 800)):
    frame_sd_map = frame.get_sd_map()
    image_bev = draw_sd_map(
        frame_sd_map, 
    )
    res_image_bev = cv2.resize(image_bev, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return res_image_bev

def load_sdmap_raster(frame, dsize=(400, 800)):
    frame_sd_map = frame.get_sd_map()
    categorical_map = rasterize_sd_map(
        frame_sd_map, 
    )
    # categorical_map is -1 where there's nothing, and 0 - 2 for roads, pedestrians, crossing
    categorical_map = cv2.resize(categorical_map, dsize=dsize, interpolation=cv2.INTER_NEAREST)

    onehot_raster_map = np.zeros((dsize[1], dsize[0], 3), dtype=np.int32)

    road_type_vals = categorical_map[categorical_map >= 0]
    vec_type_vals = vec_onehot(road_type_vals, dim=3)
    onehot_raster_map[categorical_map >= 0] = vec_type_vals

    return categorical_map, onehot_raster_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rasterized SD maps.')
    parser.add_argument('--collection', type=str, default='data_dict_sample_sd', required=False,
                        help='split to process raster SD maps.')
    parser.add_argument('--map_dir_prefix', type=str, default='olv2_centerlane', required=False)
    parser.add_argument('--total_parts', type=int, default=1, required=False)
    parser.add_argument('--part', type=int, default=0, required=False)

    args = parser.parse_args()

    root_path = '../data/OpenLane-V2'
    collection = Collection(root_path, root_path, args.collection)

    if args.total_parts > 1:
        num_frames_ppart = len(collection.frames) // args.total_parts + 1
        raster_frames = list(collection.frames.items())[num_frames_ppart * args.part: num_frames_ppart * (args.part + 1)]
    else:
        raster_frames = collection.frames.items()

    for frame_key, frame in tqdm(raster_frames):
        map_img_dir = osp.join(root_path, f"{args.map_dir_prefix}_img", frame_key[0], frame_key[1])
        os.makedirs(map_img_dir, exist_ok=True)

        map_category_dir = osp.join(root_path, f"{args.map_dir_prefix}_category_bev", frame_key[0], frame_key[1])
        os.makedirs(map_category_dir, exist_ok=True)

        map_raster_dir = osp.join(root_path, f"{args.map_dir_prefix}_raster", frame_key[0], frame_key[1])
        os.makedirs(map_raster_dir, exist_ok=True)

        # map_img_path = osp.join(map_img_dir, f"{frame_key[-1]}.png")
        map_category_path = osp.join(map_category_dir, f"{frame_key[-1]}.npz")
        # map_raster_path = osp.join(map_raster_dir, f"{frame_key[-1]}.npz")
        if osp.exists(map_category_path):
            continue
        
        # img_crop = load_sdmap_image(frame)
        cat_raster_crop, _ = load_sdmap_raster(frame, dsize=(100, 200))
        sparse_cat_raster_crop = sparse.GCXS(cat_raster_crop.astype(int))
        # sparse_oh_raster_crop = sparse.COO(oh_raster_crop)

        # save image
        # mmcv.imwrite(img_crop, map_img_path)
        sparse.save_npz(map_category_path, sparse_cat_raster_crop)
        # sparse.save_npz(map_raster_path, sparse_oh_raster_crop)
    print("Finished parsing")