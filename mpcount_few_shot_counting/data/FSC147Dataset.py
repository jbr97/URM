import os
import json
from PIL import Image
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF
import torchvision.transforms.functional as F
import ast
import torch.nn.functional as F
def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j

def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w
    
    return (left, top, right, bottom), h, w



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

class FSC147Dataset(Dataset):
    def __init__(
        self, data_path="data_dir/FSC147/", img_size=512, split='train', num_objects=3, tiling_p=0.5, zero_shot=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot

        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))

        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:

            splits = json.load(file)
            self.image_names = splits[split]

        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

        self.classes = pd.read_csv(
            os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), sep='\t', index_col=0, header=None
        )
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
        img = Image.open(os.path.join(
            self.data_path, 'images_384_VarV2', self.image_names[idx]
        )).convert("RGB")
        class_name = self.classes.loc[self.image_names[idx]]
        class_name = class_name.iloc[0]  # DONE TODO return

        w, h = img.size
        if self.split != 'train':
            img = self.transform(img)
        else:
            img1 = self.transform(img)
            img2 = self.more_transform(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]  # [3, 4]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size


        dmap = torch.from_numpy(np.load(os.path.join(
            self.data_path, 'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)
        if self.split == "train":

            bmap_orig = dmap.clone().reshape(1, dmap.shape[1]//16, 16, dmap.shape[2]//16, 16).sum(dim=(2, 4))
            bmap = (bmap_orig > 0).float()
 
            if torch.rand(1) < self.horizontal_flip_p:
                img1 = TVF.hflip(img1)
                img2 = TVF.hflip(img2)
                dmap = TVF.hflip(dmap)
                bmap = TVF.hflip(bmap)
                bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2,0]]  

            return img1, img2, bboxes, dmap, bmap
        else:
            return img,bboxes,dmap



    def __len__(self):
        return len(self.image_names)

    