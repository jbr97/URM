import os
import json
import argparse
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

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

    with open(
        os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i, (image_name, ann) in enumerate(tqdm(annotations.items())):
        _, h, w = T.ToTensor()(Image.open(os.path.join(
            data_path,
            'images_384_VarV2',
            image_name
        ))).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

        points = (
            torch.tensor(ann['points'], device=device) *
            torch.tensor([w_ratio, h_ratio], device=device)
        ).long()
        points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
        points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)
        bboxes = box_convert(torch.tensor(
            ann['box_examples_coordinates'],
            dtype=torch.float32,
            device=device
        )[:3, [0, 2], :].reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
        bboxes = bboxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=device)
        window_size = bboxes.mean(dim=0)[2:].cpu().numpy()[::-1]

        dmap = torch.zeros(*target_size)
        for p in range(points.size(0)):
            dmap[points[p, 1], points[p, 0]] += 1
        dmap = gaussian_filter(dmap.cpu().numpy(), window_size / 8)

        np.save(os.path.join(density_map_path, os.path.splitext(image_name)[0] + '.npy'), dmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument('--data_path', default='/home/ma-user/work/data/count/FSC147/')
    parser.add_argument('--image_size', type=tuple, default=(960,960))
    args = parser.parse_args()

    generate_density_maps(args.data_path, args.image_size)
