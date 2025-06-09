import os
import json
import cv2
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

class LVISDataset(Dataset):  # FSC147 format
    def __init__(
        self, data_path, img_size, split='train', num_objects=3, tiling_p=0.5, zero_shot=False
    ):
        self.split = split
        self.data_path = data_path
        self.num_objects = num_objects
        self.zero_shot = zero_shot

        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p

        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split))
        if split == 'train':
            pseudo_label_file = f"data/instances_{split}_cate.json"  # for category id
        else:
            pseudo_label_file = os.path.join(data_path, "annotations", f"instances_{split}.json")
        print('loading annotation file from', pseudo_label_file)
        self.coco = COCO(pseudo_label_file)

        self.image_ids = self.coco.getImgIds()
        print("with number of images: ", self.__len__())  # 4000 for training images

        self.img_path = os.path.join(data_path, "images")

        if split == 'train':
            self.count_anno_file = os.path.join("data", f"count_{split}_cate.json")  # for category name
        else:
            self.count_anno_file = os.path.join(data_path, "annotations", f"count_{split}.json")
        print('loading count annotation file:', self.count_anno_file)
        self.count_anno = self.load_json(self.count_anno_file)

        if split == 'train':
            self.categories = self.count_anno['categories']
            self.categories = {item['id']: item['name'] for item in self.categories}

            with open('data/lvis_4prompts_30tokens.json', 'r', encoding='utf-8') as file:
                self.descriptions = json.load(file)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[
            0]  # {'id': 2194, 'width': 500, 'height': 375, 'file_name': '000000027810.jpg'}
        img_file = img_info["file_name"]
        assert self.count_anno["images"][idx]["file_name"] == img_file
        img = Image.open(os.path.join(self.img_path, img_file)).convert("RGB")

        all_points = np.array(self.count_anno["annotations"][idx]["points"], dtype=np.float32)  # (n, 2)
        number = len(all_points)
        points = torch.from_numpy(all_points)

        w, h = img.size
        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)

            category_id = img_info['category_id']
            class_name = self.categories[category_id]

            prompts = []
            prompts.append(f'A photo a {class_name}.')
            prompts.append(f'A photo of {number} {class_name}.')
            prompts.append(f'A bad photo of a {class_name}.')
            prompts.append(f'A photo of many {class_name}.')
            prompts.append(f'A low resolution photo of the {class_name}.')
            prompts.append(f'A photo of a hard to see {class_name}.')
            prompts.append(f'A cropped photo of a {class_name}.')
            prompts.append(f'A blurry photo of a {class_name}.')
            prompts.append(f'A good photo of a {class_name}.')

            description = self.descriptions[class_name]
            description = description[::5]
            for descri in description:
                prompts.append(descri)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]  # only 5 annotated bboxes
        bboxes, wh = list(), list()
        for bbox in ex_bboxes[:3]:  # select 3 exampler bboxes
            x, y, box_w, box_h = bbox
            x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            bboxes.append([x1, y1, x2, y2])
            wh.append([box_w, box_h])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # [3, 4]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        points = points / torch.tensor([w, h]) * self.img_size

        if self.split == 'train':
            img, bboxes, clip_img, points = self.augment(img, bboxes, points)

        wh = torch.tensor(wh, dtype=torch.float32)  # [3, 2]
        wh = wh / torch.tensor([w, h]) * self.img_size
        window_size = wh.mean(dim=0).numpy()[::-1]  # (2,)

        density_map = torch.zeros(self.img_size, self.img_size)
        points[:, 0] = points[:, 0].clip(0, self.img_size - 1)
        points[:, 1] = points[:, 1].clip(0, self.img_size - 1)
        for p in range(points.size(0)):
            px, py = points[p, 0].item(), points[p, 1].item()
            px, py = int(px), int(py)
            density_map[py, px] += 1  # Fix My Name

        density_map = gaussian_filter(density_map.numpy(), window_size / 8)
        density_map = torch.from_numpy(density_map).unsqueeze(0)


        if self.split == 'train':
            return img, bboxes, density_map, class_name, clip_img, torch.from_numpy(
                np.zeros((self.img_size, self.img_size), dtype=np.uint8)), prompts
        else:
            return img, bboxes, density_map

    def augment(self, img, bboxes, points):
        if torch.rand(1) >= self.tiling_p:
            img = self.jitter(img)

        clip_img = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
        clip_img = T.Resize((224, 224), interpolation=BICUBIC)(clip_img)

        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            clip_img = TVF.hflip(clip_img)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]  # attention
            points[:, 0] = self.img_size - points[:, 0]  # FIXED
        return img, bboxes, clip_img, points

    def __len__(self):
        return len(self.image_ids)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
