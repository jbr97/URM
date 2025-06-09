import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import yaml
import argparse
from utils.object_loss import ObjectNormalizedL2Loss

from trainers.dgtrainer import DGTrainer
from models.models import DGModel_base, DGModel_mem, DGModel_memadd, DGModel_cls, DGModel_memcls, DGModel_final

from utils.misc import seed_worker, get_seeded_generator, seed_everything
from data.load_dataset import load_dataset

def get_model(name, params):

    if name == 'base':
        return DGModel_base(**params)
    elif name == 'mem':
        return DGModel_mem(**params)
    elif name == 'memadd':
        return DGModel_memadd(**params)
    elif name == 'cls':
        return DGModel_cls(**params)
    elif name == 'memcls':
        return DGModel_memcls(**params)
    elif name == 'final':
        return DGModel_final(**params)

def get_loss():
    return ObjectNormalizedL2Loss()


def get_optimizer(name, params, model):
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError('Unknown optimizer: {}'.format(name))
    
def get_optimizer(name, params, model):
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError('Unknown optimizer: {}'.format(name))    

def get_scheduler(name, params, optimizer):

    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
    else:
        raise ValueError('Unknown scheduler: {}'.format(name))

def load_config(config_path, task):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    init_params = {}
    task_params = {}

    init_params['seed'] = cfg['seed']
    init_params['version'] = cfg['version']
    init_params['device'] = cfg['device']
    init_params['log_para'] = cfg['log_para']
    init_params['patch_size'] = cfg['patch_size']
    init_params['mode'] = cfg['mode']

    seed_everything(cfg['seed'])

    task_params['model'] = get_model(cfg['model']['name'], cfg['model']['params'])

    task_params['checkpoint'] = cfg['checkpoint']

    generator = get_seeded_generator(cfg['seed'])

    if task == 'train' or task == 'train_test':
        task_params['loss'] = get_loss()

        train_set, val_set, train_loader, val_loader = load_dataset(cfg)

        task_params['train_dataloader'] = train_loader
        task_params['val_dataloader'] = val_loader
        task_params['optimizer'] = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'])
        task_params['scheduler'] = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], task_params['optimizer'])
        task_params['num_epochs'] = cfg['num_epochs']

    return init_params, task_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dg.yaml', help='path to config file')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='path to config file')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'vis'], help='task to perform')   
    args = parser.parse_args()

    init_params, task_params = load_config(args.config, args.task)
    trainer = DGTrainer(log_dir=args.log_dir,**init_params)
    os.system(f'cp {args.config} {trainer.log_dir}')

    if args.task == 'train':
        trainer.train(**task_params)
    elif args.task == 'test':
        trainer.test(**task_params)
    elif args.task == 'vis':
        trainer.vis(**task_params)
    else:
        raise ValueError('Unknown task: {}'.format(args.task))