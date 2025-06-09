import os
import json

import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map

class FSCD_LVISDataset(Dataset):
    def __init__(
        self, data_path, img_size, split='train', num_objects=3, tiling_p=0.5, zero_shot=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot

        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split))
        pseudo_label_file = "instances_" + split + ".json"
        print('loading annotation file:', os.path.join(data_path, "annotations", pseudo_label_file))
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()
        print("with number of images: ", self.__len__())  

        self.img_path = os.path.join(data_path, "images")
        self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        print('loading count annotation file:', self.count_anno_file)
        self.count_anno = self.load_json(self.count_anno_file)

        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))

        self.more_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx: int):
        if self.split == 'train':
            return self.get_train_item(idx)
        else:
            return self.get_val_item(idx)

    def get_train_item(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0] 
        img_file = img_info["file_name"]
        assert self.count_anno["images"][idx]["file_name"] == img_file

        img = Image.open(os.path.join(self.img_path, img_file)).convert("RGB")
        raw = np.array(img).copy()
        w, h = img.size
        img1 = self.transform(img)
        img2 = self.more_transform(img)


        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]  # only 5 annotated bboxes
        bboxes, wh = list(), list()
        for bbox in ex_bboxes[:3]:  # select 3 exampler bboxes
            x, y, box_w, box_h = bbox
            x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            bboxes.append([x1, y1, x2, y2])
            wh.append([box_w, box_h])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # [3, 4]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        wh = torch.tensor(wh, dtype=torch.float32)  # [3, 2]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        window_size = wh.mean(dim=0).numpy()[::-1]  # (2,)

        all_points = np.array(self.count_anno["annotations"][idx]["points"], dtype=np.float32)  # (n, 2)
        points = torch.from_numpy(all_points)
        density_map = torch.zeros(h, w)
        for p in range(points.size(0)):
            px, py = points[p, 1].item(), points[p, 0].item()
            px, py = int(px), int(py)
            density_map[px, py] += 1
        density_map = gaussian_filter(density_map.numpy(), window_size / 8)
        density_map = torch.from_numpy(density_map).unsqueeze(0)

        original_sum = density_map.sum()
        density_map = self.resize(density_map)
        density_map = density_map / density_map.sum() * original_sum

        bmap_orig = density_map.clone().reshape(1, density_map.shape[1]//16, 16, density_map.shape[2]//16, 16).sum(dim=(2, 4))
        bmap = (bmap_orig > 0).float()

        if torch.rand(1) < self.horizontal_flip_p:
            img1 = TVF.hflip(img1)
            img2 = TVF.hflip(img2)
            density_map = TVF.hflip(density_map)
            bmap = TVF.hflip(bmap)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]  # attention


        return img1, img2, bboxes, density_map, bmap

    def get_val_item(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]  
        img_file = img_info["file_name"]
        assert self.count_anno["images"][idx]["file_name"] == img_file

        img = Image.open(os.path.join(self.img_path, img_file)).convert("RGB")
        w, h = img.size
        origin_w, origin_h = w, h
        img = T.ToTensor()(img)
        if w % 32 != 0:
            w = int(w / 32) * 32 + 32
        if h % 32 != 0:
            h = int(h / 32) * 32 + 32
        pad_w, pad_h = w - origin_w, h - origin_h
        img = nn.ZeroPad2d((0, pad_w, 0, pad_h))(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]  # only 5 annotated bboxes
        bboxes, wh = list(), list()
        for bbox in ex_bboxes[:3]:  # select 3 exampler bboxes
            x, y, box_w, box_h = bbox
            x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            bboxes.append([x1, y1, x2, y2])
            wh.append([box_w, box_h])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # [3, 4]
        wh = torch.tensor(wh, dtype=torch.float32)  # [3, 2]
        window_size = wh.mean(dim=0).numpy()[::-1]  # (2,)

        all_points = np.array(self.count_anno["annotations"][idx]["points"], dtype=np.float32)  # (n, 2)
        points = torch.from_numpy(all_points)
        density_map = torch.zeros(h, w)
        for p in range(points.size(0)):
            px, py = points[p, 1].item(), points[p, 0].item()
            px, py = int(px), int(py)
            density_map[px, py] += 1
        density_map = gaussian_filter(density_map.numpy(), window_size / 8)
        density_map = torch.from_numpy(density_map).unsqueeze(0)
        return img, bboxes, density_map

    def __len__(self):
        return len(self.image_ids)
    
    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
