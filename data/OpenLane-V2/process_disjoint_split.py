import pickle
import os.path as osp
from openlanev2.centerline.io import io
from openlanev2.centerline.preprocessing import collect
from openlanev2.centerline.dataset import Collection, Frame

def collect_from_keys(root_path, keys_file_collection, train_frames, val_frames):
    r"""
    Collect data from keys_file, and store in a .pkl with split as file name.

    Parameters
    ----------
    root_path : str
    keys_file : str
        File name of the keys file.

    """
    data_dict = {}
    with open(f'{root_path}/{keys_file_collection}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            frame_key = tuple(line.split())
            if frame_key[0] == 'train':
                data_dict[frame_key] = train_frames[frame_key]
            else:  # val
                data_dict[frame_key] = val_frames[frame_key]

    io.pickle_dump(f'{root_path}/{keys_file_collection}.pkl', data_dict)

if __name__ == '__main__':
    root_path = './OpenLane-V2'
    disj_train_split_keys_file = 'data_dict_subset_A_train_disjoint'
    disj_val_split_keys_file = 'data_dict_subset_A_val_disjoint'


    openlaneV2_train_frames = pickle.load(open(osp.join(root_path, f"data_dict_subset_A_train.pkl"), "rb"))
    openlaneV2_val_frames = pickle.load(open(osp.join(root_path, f"data_dict_subset_A_val.pkl"), "rb"))

    collect_from_keys(root_path, disj_train_split_keys_file, openlaneV2_train_frames, openlaneV2_val_frames)
    collect_from_keys(root_path, disj_val_split_keys_file, openlaneV2_train_frames, openlaneV2_val_frames)
