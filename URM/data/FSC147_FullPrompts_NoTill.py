import os
import json
import cv2
from PIL import Image
import numpy as np
import pandas as pd

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC

except ImportError:
    BICUBIC = Image.BICUBIC

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

from scipy.ndimage import gaussian_filter

class FSC147Dataset(Dataset):
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

        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]

        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

        if split == 'train':
            self.classes = pd.read_csv(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), sep='\t', index_col=0, header=None
            )

            with open('data/FSC147_4prompt_30tokens.json', 'r', encoding='utf-8') as file:
                self.descriptions = json.load(file)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(
            self.data_path, 'images_384_VarV2', self.image_names[idx]
        )).convert("RGB")

        points = self.annotations[self.image_names[idx]]['points']
        number = len(points)

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

            class_name = self.classes.loc[self.image_names[idx]]
            class_name = class_name.iloc[0]

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

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]  # [3, 4]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        points = torch.tensor(points)  # [n, 2]
        points = points / torch.tensor([w, h]) * self.img_size

        if self.split == 'train':
            img, bboxes, clip_img, points = self.augment(img, bboxes, points)

            density_map = self.generate_dmap(points, bboxes)
            density_map = torch.from_numpy(density_map).unsqueeze(0)

            return img, bboxes, density_map, class_name, clip_img, torch.from_numpy(
                np.zeros((self.img_size, self.img_size), dtype=np.uint8)), prompts
        else:
            density_map = torch.from_numpy(np.load(os.path.join(
                self.data_path, 'gt_density_map_adaptive_512_512_object_VarV2',
                os.path.splitext(self.image_names[idx])[0] + '.npy',
            ))).unsqueeze(0)
            return img, bboxes, density_map

    def generate_dmap(self, points, bboxes):
        points[:, 0] = points[:, 0].clip(0, self.img_size - 1)
        points[:, 1] = points[:, 1].clip(0, self.img_size - 1)

        window_size = bboxes.mean(dim=0)[2:].numpy()  # (2,)
        window_size = window_size[::-1]  # (2,)

        dmap = torch.zeros((self.img_size, self.img_size))
        for p in range(points.size(0)):
            dmap[int(points[p, 1].item()), int(points[p, 0].item())] += 1

        dmap = gaussian_filter(dmap.numpy(), window_size / 8)
        return dmap

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
            points[:, 0] = self.img_size - points[:, 0]
        return img, bboxes, clip_img, points

    def __len__(self):
        return len(self.image_names)
