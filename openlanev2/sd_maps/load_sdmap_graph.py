import os
import argparse
import pickle

import os.path as osp
import numpy as np
import osmnx as ox
import av2.geometry.utm as geo_utils

# from openlanev2.io import io
from openlanev2.centerline.dataset import Collection
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pyproj import Transformer

from sd_map_utils import bbox_rot_from_point


HIGHWAY_SETS = {
    'truck_road': ['motorway', 'trunk', 'motorway_link', 'trunk_link'],
    'highway': ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link'],
    'residential': ['residential', 'living_street'],
    'service': ['service'],
    'pedestrian': ['pedestrian', 'footway', 'path', 'steps'],
    'road': ['road'],
    'bus_way': ['busway', 'bus_guideway']
}
TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def get_xy_origin_offset(location_point):
    lat, lon = location_point

    res = TRAN_4326_TO_3857.transform(lon, lat)
    # gdf_o = gpd.GeoDataFrame(geometry=[shapely.geometry.Point([lon, lat])], crs="EPSG:4326")
    # to = gdf_o.to_crs(crs=3857)
    # xo,yo = to.loc[0,"geometry"].xy
    return res

def parse_graph(G_simplify, xo, yo, rot_deg):
    if not G_simplify.edges:
        # no edges
        graph_dict = {}
        for lanetype in HIGHWAY_SETS.keys():
            graph_dict[lanetype] = []
    
    else:
        # get dataframe
        gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G_simplify, node_geometry=True, fill_edge_geometry=True)

        gdf_save = gdf_edges[['highway', 'geometry']].copy()

        # parse highway type
        gdf_save = gdf_save.explode('highway')

        highway_num = gdf_save.highway.copy()
        union_highway_set = set([tag for _, v in HIGHWAY_SETS.items() for tag in v])
        for k, v in HIGHWAY_SETS.items():
            highway_num[gdf_save.highway.isin(v)] = k
        highway_num[~gdf_save.highway.isin(union_highway_set)] = "other"  # catch all
        gdf_save["HIGHWAY_TYPE"] = highway_num #.astype(int)

        # transform geometry to local coordinate
        geo_ = gdf_save.geometry.to_crs(crs=3857)
        geo_ = geo_.translate(xoff=-xo, yoff=-yo)
        geo_ = geo_.rotate(-rot_deg[0], origin=(0, 0))
        gdf_save['geometry'] = geo_
        
        # get dictionary form
        graph_dict = {}
        for lanetype in HIGHWAY_SETS.keys():
            polylines = list(gdf_save[gdf_save['HIGHWAY_TYPE'] == lanetype].geometry.apply(lambda x: np.stack(x.coords.xy,axis=-1)))
            graph_dict[lanetype] = polylines
        # add one for catch all
        polylines = list(gdf_save[gdf_save['HIGHWAY_TYPE'] == "other"].geometry.apply(lambda x: np.stack(x.coords.xy,axis=-1)))
        graph_dict["other"] = polylines
    return graph_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rasterized SD maps.')
    parser.add_argument('--collection', type=str, default='data_dict_sample_train', required=False,
                        help='split to process raster SD maps.')
    parser.add_argument('--city_names', type=str, default='train_city', required=False)
    # "sd_map"
    parser.add_argument('--map_dir_prefix', type=str, default='sd_map_graph_all', required=False)
    parser.add_argument('--threshold', type=float, default=0.1, required=False)
    parser.add_argument('--total_parts', type=int, default=1, required=False)
    parser.add_argument('--part', type=int, default=0, required=False)

    args = parser.parse_args()

    root_path = '../data/OpenLane-V2'
    city_dict = pickle.load(open(f"../data/ArgoverseV2/{args.city_names}.pkl", "rb"))
    collection = Collection(root_path, root_path, args.collection)

    if args.total_parts > 1:
        num_frames_ppart = len(collection.frames) // args.total_parts + 1
        raster_frames = list(collection.frames.items())[num_frames_ppart * args.part: num_frames_ppart * (args.part + 1)]
    else:
        raster_frames = collection.frames.items()

    for frame_key, frame in tqdm(raster_frames):
        map_crop_dir = osp.join(root_path, args.map_dir_prefix, frame_key[0], frame_key[1])
        os.makedirs(map_crop_dir, exist_ok=True)

        map_crop_path = osp.join(map_crop_dir, f"{frame_key[-1]}.pkl")
        if osp.exists(map_crop_path):
            continue

        city_name = city_dict[frame.meta['meta_data']['source_id']]

        points = np.array([frame.get_pose()['translation'][:-1]])
        wgs_pose = geo_utils.convert_city_coords_to_wgs84(points, city_name=city_name)
        location_point = wgs_pose[0,]
        xo, yo = get_xy_origin_offset(location_point)

        r = R.from_matrix(frame.get_pose()['rotation'])
        rot_deg = r.as_euler('zyx', degrees=True)

        north, south, east, west, bbox = bbox_rot_from_point(location_point, dist_x=80, dist_y=100, rot_deg=rot_deg[0], return_poly=True)

        # Pull graph from OpenStreetMap
        G = ox.graph_from_bbox(
                north,
                south,
                east,
                west,
                network_type="all",
                simplify=False,
                retain_all=True,
                truncate_by_edge=False,
                clean_periphery=True,
        )

        G_simplify = ox.simplify_graph(G)

        # Parse SD map into a graph, relative to ego-pose. Store into dictionary.
        graph_dict = parse_graph(G_simplify, xo, yo, rot_deg)
        
        pickle.dump(graph_dict, open(map_crop_path, "wb"))

