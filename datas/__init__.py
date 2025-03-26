import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader
from tllib.utils.data import ForeverDataIterator

from configs import CFG
from datas.datasets import HyRankDataset, ShangHangDataset


def build_transform():
    if CFG.DATASET.NAME in ['HyRANK', 'ShangHang']:
        transform = transforms.Compose([
            transforms.LabelRenumber(),
            transforms.ZScoreNormalize(),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    assert split in ['train', 'val', 'test', 'dynamic']
    if CFG.DATASET.NAME == 'HyRANK':
        dataset = HyRankDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                CFG.DATASET.PATCH.PAD_MODE, None,
                                None, transform=build_transform())
    elif CFG.DATASET.NAME == 'ShangHang':
        dataset = ShangHangDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                   CFG.DATASET.PATCH.PAD_MODE, None,
                                   None, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


def build_dataloader(dataset, sampler=None, drop_last=True):
    try:
        nodes = dist.get_world_size()
    except:
        nodes = 1
    return DataLoader(dataset,
                      batch_size=CFG.DATALOADER.BATCH_SIZE // nodes,
                      num_workers=CFG.DATALOADER.NUM_WORKERS,
                      pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                      sampler=sampler,
                      drop_last=drop_last
                      )


def build_iterator(dataloader: DataLoader):
    return ForeverDataIterator(dataloader)

