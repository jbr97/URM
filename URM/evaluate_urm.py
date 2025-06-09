from urm.loca import build_model

from utils.arg_parser import get_argparser
from data.load_dataset import load_dataset
import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


@torch.no_grad()
def evaluate(args):

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    _, test, _, test_loader = load_dataset(args)
    from utils.losses import ObjectNormalizedL2Loss
    criterion = ObjectNormalizedL2Loss()
    test_ae = torch.tensor(0.0).to(device)
    test_mse = torch.tensor(0.0).to(device)
    test_loss = torch.tensor(0.0).to(device)
    model.eval()
    for img, bboxes, density_map in test_loader:
        img = img.to(device)
        bboxes = bboxes.to(device)
        density_map = density_map.to(device)

        with torch.no_grad():
            out, aux_out, _, _ = model(img, bboxes)
            num_objects = density_map.sum()
            dist.all_reduce_multigpu([num_objects])

        main_loss = criterion(out, density_map, num_objects)
        test_loss += main_loss * img.size(0)
        test_ae += torch.abs(
            density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
        ).sum()
        test_mse += torch.pow(
            density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1), 2
        ).sum()

    dist.all_reduce_multigpu([test_loss])
    dist.all_reduce_multigpu([test_ae])
    if rank == 0:
        rmse = test_mse / len(test)
        rmse = torch.sqrt(rmse).item()
        print(
            f"test set ",
            f"MAE: {test_ae.item() / len(test):.3f}",
            f"RMSE: {rmse:.3f}",
        )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)
