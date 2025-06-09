import os
import cv2
import numpy as np
from tqdm import tqdm

path = '/home/ma-user/work/cxn5/dataset/FSC147/gt_density_map_adaptive_512_512_object_VarV2'
save_path = '/home/ma-user/work/cxn5/dataset/FSC147/vis_dmap_max'
os.makedirs(save_path, exist_ok=True)
dmaps = os.listdir(path)
for dmap_name in tqdm(dmaps):
    dmap_path = os.path.join(path, dmap_name)
    dmap = np.load(dmap_path)  # (512, 512)
    dmap = dmap / dmap.max()
    dmap *= 255
    cv2.imwrite(os.path.join(save_path, dmap_name.replace('npy', 'jpeg')), dmap)
