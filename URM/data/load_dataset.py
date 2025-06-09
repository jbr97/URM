from torch.utils.data import DataLoader, DistributedSampler

def load_dataset(args):
    print('Loading train data from:', args.data_path)
    train_set, val_set, train_loader, val_loader = None, None, None, None
    if args.data == 'FSC147_FullPrompts_NoTill':
        from data.FSC147_FullPrompts_NoTill import FSC147Dataset
        train_set = FSC147Dataset(
            args.data_path, args.image_size, split='train',
            num_objects=args.num_objects, tiling_p=args.tiling_p, zero_shot=args.zero_shot
        )
    elif args.data == 'LVIS_FullPrompts_NoTill':
        from data.LVISDataset_FullPrompts_NoTill import LVISDataset
        train_set = LVISDataset(
            args.data_path, args.image_size, split='train',
            num_objects=args.num_objects, tiling_p=args.tiling_p, zero_shot=args.zero_shot
        )
    else:
        train_set = None

    print('Loading test data from:', args.val_data_path)
    if args.val_data == 'LVIS_FullPrompts_NoTill':
        from data.LVISDataset_FullPrompts_NoTill import LVISDataset
        val_set = LVISDataset(
            args.val_data_path, args.val_image_size, split='test',
            num_objects=args.num_objects, zero_shot=args.zero_shot
        )
    elif args.val_data == 'FSC147_FullPrompts_NoTill':
        from data.FSC147_FullPrompts_NoTill import FSC147Dataset
        val_set = FSC147Dataset(
            args.val_data_path, args.val_image_size, split='val',
            num_objects=args.num_objects, zero_shot=args.zero_shot
        )
    else:
        val_set = None
    if train_set:
        train_loader = DataLoader(
            train_set, sampler=DistributedSampler(train_set), batch_size=args.batch_size,
            drop_last=True, num_workers=args.num_workers
        )
    if val_set:
        val_loader = DataLoader(
            val_set, sampler=DistributedSampler(val_set), batch_size=args.val_batch_size,
            drop_last=False, num_workers=args.num_workers
        )
    return train_set, val_set, train_loader, val_loader
