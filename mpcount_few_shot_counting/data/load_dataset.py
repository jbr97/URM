from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset

from data.FSC147Dataset import FSC147Dataset
from data.FSCD_LVISDataset import FSCD_LVISDataset
import torch


def collate(batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0)
        images2 = torch.stack(transposed_batch[1], 0)
        bboxes = torch.stack(transposed_batch[2],0)  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[3], 0)
        bmaps = torch.stack(transposed_batch[4], 0)
        return images1, images2, (bboxes, dmaps, bmaps)

def collate_val(batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0)
        bboxes = torch.stack(transposed_batch[1],0)  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        return images1,bboxes,dmaps

def load_dataset(args):
    print('Loading train data from:', args['data_path'])
    if args['data'] == 'FSC147':
        train_set = FSC147Dataset(
            args['data_path'], args['image_size'], split='train',
            num_objects=args['num_objects'], tiling_p=args['tiling_p'], 
        )
    elif args['data']  == 'FSCD_LVIS':
        train_set = FSCD_LVISDataset(
            args['data_path'], args['image_size'], split='train',
            num_objects=args['num_objects'], tiling_p=args['tiling_p'], 
        )

    print('Loading test data from:', args['val_data_path'])
    if args['val_data']  == 'FSC147':
        val_set = FSC147Dataset(
            args['val_data_path'], args['val_image_size'], split='val',
            num_objects=args['num_objects'], tiling_p=args['tiling_p']
        )
    elif args['val_data'] == 'FSCD_LVIS':
        val_set = FSCD_LVISDataset(
            args['val_data_path'], args['val_image_size'],split='test',
            num_objects=args['num_objects'], tiling_p=args['tiling_p'], 
        )
    
    train_loader = DataLoader(
        train_set,  batch_size=args['train_loader']['batch_size'],collate_fn=collate,
        drop_last=True, num_workers=args['train_loader']['num_workers']
    )
    val_loader = DataLoader(
        val_set,  batch_size=args['val_loader']['batch_size'],collate_fn=collate_val,
        drop_last=False, num_workers=args['val_loader']['num_workers']
    )
    
    return train_set, val_set, train_loader, val_loader
