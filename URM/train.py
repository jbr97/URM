import os
import argparse
import random
import numpy as np
from time import perf_counter

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.arg_parser import get_argparser
from data.load_dataset import load_dataset
from utils.losses import ObjectNormalizedL2Loss

def load_optimizer(args, model):
    print('lr:', args.lr, 'backbone lr:', args.backbone_lr, 'weight decay:', args.weight_decay)
    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:  # only include learnable params
            continue

        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    return optimizer

def load_clip(name):
    import clip
    from clip import tokenize
    model, preprocess = clip.load(name)
    return model, preprocess, tokenize

def load_model(args):
    from urm.loca import build_model
    model = DistributedDataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu, find_unused_parameters=True
    )

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_num = total_num - trainable_num
    print('\nParams:', total_num / 1e6)
    print('Trainable Params:', trainable_num / 1e6, 'Un-trainble Params:', untrainable_num / 1e6)
    return model

def train(args, save_path, device):
    if args.clip:
        print('\nLoading CLIP:', args.clip)
        clip_model, clip_preprocess, tokenize = load_clip(args.clip)
        print('Loading proj by MaskCLIP way')
        clip_model.load_visual_projs()
        print('CLIP visual feature mode:', args.clip_feat)

    print('\nLoading model...')
    model = load_model(args)

    print('\nLoading optimizer')
    optimizer = load_optimizer(args, model)

    print('Loading scheduler:', args.scheduler)
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    else:
        cosine_with_warmup_lr = lambda epoch: 0.1 + (epoch / args.warmup) if epoch <= args.warmup else \
            0.5 * (np.cos((epoch - args.warmup) / (args.epochs - args.warmup) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_warmup_lr)

    criterion = ObjectNormalizedL2Loss()
    l1_loss = nn.L1Loss()

    start_epoch = 0
    best = 1e10
    best_rmse = 1e10
    if args.resume_training:
        print('\nResume from', args.model_path)
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        best_rmse = checkpoint['bets_val_rmse']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print('\nLoading data:', args.data, args.val_data)
    train_set, val_set, train_loader, val_loader = load_dataset(args)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print('\nStart epoch', epoch)
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        train_distill_langu_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        val_mse = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)

        model.train()
        for iter, (img, bboxes, density_map, class_names, clip_img, _, prompt) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            clip_img = clip_img.to(device)

            if args.clip:
                with torch.no_grad():
                    ''' text tower '''
                    if isinstance(prompt, tuple):  # single vanilla prompt
                        token = tokenize(prompt).to(device)
                        text_feat = clip_model.encode_text(token)
                    else:  # full prompts
                        bs = len(img)
                        text_feats = []
                        for b in range(bs):
                            tokens = []
                            # for each prompt in current batch
                            for k in range(len(prompt)):
                                token = tokenize(prompt[k][b]).to(device)  # [1, 77]
                                tokens.append(token)
                            tokens = torch.cat(tokens)  # [9, 77]

                            curr_text_feat = clip_model.encode_text(tokens)  # [9, 512]
                            curr_text_feat = torch.mean(curr_text_feat, dim=0).unsqueeze(0)  # [1, 512]

                            text_feats.append(curr_text_feat)
                        text_feat = torch.cat(text_feats)  # [b, 512]

                    ''' vision tower with MaskCLIP operation '''

                    seg, local_feat, global_feat = clip_model.encode_image_feature(clip_img, text_feat)  # [b, 1, 14, 14], [b, 512, 14, 14], [b, 512, 14, 14]

                    ''' segmentation '''
                    mean = torch.mean(seg, dim=(1, 2, 3), keepdim=True)  # [b, 1, 1, 1]
                    std = torch.std(seg, dim=(1, 2, 3), keepdim=True)
                    seg = (seg - mean) / (std + 1e-8)
                    threshold = 0.65
                    seg[seg <= threshold] = 0.
                    seg[seg > threshold] = 1.

                    ''' mask pool '''
                    img_feat = global_feat * seg  # [b, 512, 14, 14]

                    img_feat = img_feat.reshape(img_feat.shape[0], img_feat.shape[1], -1)  # [b, 512, 196]
                    img_feat = torch.sum(img_feat, dim=-1)  # [b, 512]
                    norm = torch.sum(seg, dim=(-3, -2, -1), keepdims=False)  # [b]
                    norm = norm.unsqueeze(1)  # [b, 1]
                    img_feat = img_feat / (norm + 1e-8)

                    vis = False
                    if vis:
                        print(iter)
                        save_path = os.path.join(f'./vis_segment_{threshold}/')
                        os.makedirs(save_path, exist_ok=True)
                        mean, std = torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711])
                        mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)
                        vis_img, vis_feature = clip_img.detach(), seg.detach()
                        vis_feature = F.interpolate(vis_feature, scale_factor=8, mode='bilinear', align_corners=True)
                        vis_img, vis_feature = vis_img[0].cpu(), vis_feature[0].cpu()
                        vis_img = (vis_img * std) + mean
                        vis_feature = vis_feature / vis_feature.max()
                        vis_img, vis_feature = vis_img.permute(1, 2, 0).numpy(), vis_feature.permute(1, 2, 0).numpy()
                        vis_img, vis_feature = vis_img * 255, vis_feature * 255
                        import cv2
                        cv2.imwrite(os.path.join(save_path, str(iter) + '_img.jpg'), vis_img.astype(int))
                        cv2.imwrite(os.path.join(save_path, str(iter) + '_seg.jpg'), vis_feature.astype(int))
                out, aux_out, vision_repre, language_repre = model(img, bboxes)
            else:
                out, aux_out, _, _ = model(img, bboxes)

            with torch.no_grad():
                num_objects = density_map.sum()  # obtain the number of objects in batch
                dist.all_reduce_multigpu([num_objects])

            main_loss = criterion(out, density_map, num_objects)

            if len(aux_out) > 0:
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
            else:
                aux_loss = 0.
            loss = main_loss + aux_loss

            if args.clip:
                if isinstance(vision_repre, list):
                    distill_vision_loss = sum([
                        args.vision_distill_weight * l1_loss(img_feat, vis_repre)
                        for vis_repre in vision_repre
                    ])  # [b, 512]
                else:
                    distill_vision_loss = l1_loss(img_feat, vision_repre)
                loss = loss + args.vision_distill_weight * distill_vision_loss

                if isinstance(language_repre, list):
                    distill_language_loss = sum([
                        args.language_distill_weight * l1_loss(text_feat, lg_repre)
                        for lg_repre in language_repre
                    ])
                else:
                    distill_language_loss = l1_loss(text_feat, language_repre)
                loss = loss + args.language_distill_weight * distill_language_loss
            else:
                distill_vision_loss, distill_language_loss = 0., 0.

            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_distill_langu_loss += distill_language_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            if iter % args.print_iter == 0:
                print('### {}/{} | loss={:.4f} ###'.format(iter, len(train_loader), main_loss.item()))

            # break
        print('Validating ...')
        model.eval()
        with torch.no_grad():

            for img, bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)

                with torch.no_grad():
                    out, aux_out, _, _ = model(img, bboxes)
                    num_objects = density_map.sum()
                    dist.all_reduce_multigpu([num_objects])

                main_loss = criterion(out, density_map, num_objects)
                val_loss += main_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()
                val_mse += torch.pow(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1), 2
                ).sum()
        dist.all_reduce_multigpu([train_loss])
        dist.all_reduce_multigpu([val_loss])
        dist.all_reduce_multigpu([aux_train_loss])
        dist.all_reduce_multigpu([train_distill_langu_loss])
        dist.all_reduce_multigpu([train_ae])
        dist.all_reduce_multigpu([val_ae])

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            rmse = val_mse / len(val_set)
            rmse = torch.sqrt(rmse).item()

            if val_ae.item() / len(val_set) < best:
                best = val_ae.item() / len(val_set)
                best_rmse = rmse
                checkpoint = {
                    'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'best_val_ae': val_ae.item() / len(val_set),
                    'bets_val_rmse': best_rmse
                }
                torch.save(checkpoint, os.path.join(save_path, f'{args.model_name}.pt'))

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item() / len(train_loader):.3f}",
                f"Aux train loss: {aux_train_loss.item() / len(train_loader):.3f}",
                f"Train distill language loss: {train_distill_langu_loss.item() / len(train_loader):.3f}",
                f"Val loss: {val_loss.item() / len(val_loader):.3f}",
                f"Train MAE: {train_ae.item() / len(train_set):.3f}",
                f"Val MAE: {val_ae.item() / len(val_set):.3f}",
                f"Best MAE: {best:.3f}",
                f"Best RMSE: {best_rmse:.3f}",
                f"Epoch time: {end - start:.3f} seconds",
            )

        if epoch > args.early_stop_epoch:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()

    if args.val_batch_size == -1:
        args.val_batch_size = args.batch_size

    if args.image_size == -1:
        args.batch_size = 1
    if args.val_image_size == -1:
        args.val_batch_size = 1
    print(args.data, 'train image size:', args.image_size, 'train batch size:', args.batch_size)
    print(args.val_data, 'val image size:', args.val_image_size, 'val batch size:', args.val_batch_size)

    if args.data == 'FSCD_LVIS':
        print('WARNING: NOT Support Yet')

    print('args:', args.__dict__)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    save_path = os.path.join('logs', args.exp)
    print('save_path:', save_path)
    os.makedirs(save_path, exist_ok=True)

    train(args, save_path, device)

    dist.destroy_process_group()
