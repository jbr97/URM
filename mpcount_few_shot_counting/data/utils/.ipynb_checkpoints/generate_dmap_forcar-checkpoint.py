import os
import json
import argparse
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import glob
import torch
from torchvision.ops import box_convert
from torchvision import transforms as T

def generate_density_maps(data_path, target_size=(512, 512)):

    density_map_path = os.path.join(
        data_path,
        f'gt_density_map_adaptive_{target_size[0]}_{target_size[1]}_object_VarV2'
    )
    if not os.path.isdir(density_map_path):
        os.makedirs(density_map_path)

    anno_path = os.path.join(data_path,"Annotations")
    files = os.listdir(anno_path)
    

    device = torch.device('cpu')
    
    for file in tqdm(files):
        if not file.endswith(".txt"):
            continue
        item_annos = []
        with open(os.path.join(anno_path, file)) as f:
            lines = f.readlines()
            for line in lines:
                infos = list(map(float, line.split(" ")))
                x1, y1, x2, y2, cls_id = infos
                item_annos.append([x1,y1,x2,y2])
        
        bboxes = torch.tensor(item_annos)
        _, h, w = T.ToTensor()(Image.open(os.path.join(
            data_path,
            'Images',
            file.replace(".txt",".png")
        ))).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

        points_xy = torch.zeros((len(bboxes),2))
        points_xy[:,0] = (bboxes[:,1] + bboxes[:,3])/2
        points_xy[:,1] = (bboxes[:,0] + bboxes[:,2])/2
        points = (
            points_xy.to(device) *
            torch.tensor([w_ratio, h_ratio], device=device)
        ).long()
        points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
        points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)

        bboxes = box_convert(bboxes,in_fmt='xyxy', out_fmt='xywh')
        
        bboxes = bboxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=device)
        window_size = bboxes.mean(dim=0)[2:].cpu().numpy()[::-1]

        dmap = torch.zeros(*target_size)
        for p in range(points.size(0)):
            dmap[points[p, 1], points[p, 0]] += 1
        dmap = gaussian_filter(dmap.cpu().numpy(), window_size / 8)

        np.save(os.path.join(density_map_path, file.replace('.txt','.npy')), dmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument('--data_path', default='/home/ma-user/work/data/count/CARPKDEV/')
    parser.add_argument('--image_size', type=tuple, default=(736,1280))
    args = parser.parse_args()

    generate_density_maps(args.data_path, args.image_size)
