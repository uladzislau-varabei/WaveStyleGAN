import copy
import os
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from logger import log_message
from shared_utils import get_distributed_world_size, get_total_batch_size, is_running_on_linux


# ----- Utils -----

def load_image(img_path, img_channels):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2 and img_channels == 3:
        img = img[..., None]
    img = img.astype(np.uint8)
    return img


def create_corner_points(coords_xyxy):
    x1, y1, x2, y2 = coords_xyxy
    points = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)
    return points


def get_center_transform(src_size_yx, dst_size_yx):
    src_h, src_w = src_size_yx
    dst_h, dst_w = dst_size_yx
    scale = max(src_h / dst_h, src_w / dst_w)
    h1, h2 = 0.5 * (dst_h - src_h / scale), 0.5 * (dst_h + src_h / scale)
    w1, w2 = 0.5 * (dst_w - src_w / scale), 0.5 * (dst_w + src_w / scale)
    src_points = create_corner_points((0, 0, src_w - 1, src_h - 1))
    # print(f'src points xyxy: {w1}, {h1}, {w2}, {h2}')
    dst_points = create_corner_points((w1, h1, w2 - 1, h2 - 1))
    # print(f'dst points xyxy: {0}, {0}, {dst_w}, {dst_h}')
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


def resize_img(img, target_size_yx, resize_method=None):
    # Warp image into center of frame
    src_size_yx = img.shape[:2]
    M = get_center_transform(src_size_yx, target_size_yx)
    assert resize_method in ['area', 'cubic', 'linear', None]
    if resize_method is None:
        if (src_size_yx[0] > target_size_yx[0]) and (src_size_yx[1] > target_size_yx[1]):
            # Downsampling => INTER_AREA gives the best quality
            interpolation = cv2.INTER_AREA
            # print("Downsampling")
        else:
            # Upsampling => INTER_CUBIC gives the best quality
            interpolation = cv2.INTER_CUBIC
            # print('Upsanpling')
    else:
        interpolation = {
            'area': cv2.INTER_AREA,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR
        }[resize_method]
    img = cv2.warpPerspective(img, M, (target_size_yx[1], target_size_yx[0]),
                              flags=interpolation,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    img = img.astype(np.uint8)
    return img


def random_crop(img, target_size_yx):
    h, w = img.shape[:2]
    size_y, size_x = target_size_yx
    assert h >= size_y and w >= size_x
    start_x = np.random.randint(0, max(w - size_x, 1))
    start_y = np.random.randint(0, max(h - size_y, 1))
    end_x = start_x + size_x
    end_y = start_y + size_y
    img_crop = img[start_y : end_y, start_x : end_x]
    return img_crop


def scale_img(img, target_size_yx):
    h, w = img.shape[:2]
    size_y, size_x = target_size_yx
    assert h >= size_y and w >= size_x
    scale = 1. / min(h / size_y, w / size_x)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img = img.astype(np.uint8)
    return img


def random_crop_v1(img, target_size_yx):
    """
    Extracts crop of an image.
    1. Random crop in image full resolution
    2. Scale image, so that min size equals target size and then get random crop
    """
    if np.random.choice([True, False], p=[0.5, 0.5]):
        # Full random crop
        img = random_crop(img, target_size_yx)
    else:
        # Min scaling with preserving aspect ratio
        img = scale_img(img, target_size_yx)
        # Full random crop
        img = random_crop(img, target_size_yx)
    img_size_yx = img.shape[:2]
    assert img_size_yx[0] == target_size_yx[0] and img_size_yx[1] == target_size_yx[1]
    return img


def center_crop(img, target_size_yx):
    h, w, c = img.shape
    size_y, size_x = target_size_yx
    assert h >= size_y and w >= size_x
    cx, cy = w // 2, h // 2
    start_x, end_x = int(cx - size_x / 2), int(cx + size_x / 2)
    start_y, end_y = int(cy - size_y / 2), int(cy + size_y / 2)
    img_crop = img[start_y : end_y, start_x : end_x]
    return img_crop


def center_crop_v2(img, target_size_yx):
    # Scale and then apply center crop
    img = scale_img(img, target_size_yx)
    img = center_crop(img, target_size_yx)
    return img


def get_num_workers(num_workers, rank=0, logger=None):
    world_size = get_distributed_world_size()
    config_num_workers = copy.deepcopy(num_workers)
    if isinstance(num_workers, str):
        num_workers = num_workers.lower()
        if num_workers == 'auto':
            if world_size > 1:
                # For distributed training it's better to use this value
                num_workers = 0
            else:
                num_workers = mp.cpu_count()
                # Maybe divide by 2 for modern cpus due to hyper-threading?
                num_workers = num_workers // 2
        else:
            assert False, 'Not supported mode for num_workers'
    message = f'Dataloader: num_workers={num_workers}, config_num_workers={config_num_workers}, world_size={world_size}'
    log_message(message, rank, logger)
    return num_workers


def update_path_for_wsl(path):
    # Update path for WSL. Drives are mounted. Example: D:\\Dev -> /mnt/d/Dev
    path = path.replace('\\', '/')
    drive, path = path.split(':/', 1)
    path = os.path.join('/mnt', drive.lower(), path)
    return path

def update_df_for_os(df):
    if is_running_on_linux():
        print('Updating df paths for WSL...')
        df['path'] = df['path'].apply(lambda p: update_path_for_wsl(p))
    return df


# ----- Base dataset -----

class BaseDataset(Dataset):
    def __init__(self, config):
        general_params = config['general_params']
        dataset_params = config['dataset_params']
        csv_path = dataset_params['csv_path']
        df_preprocess = dataset_params.get('df_preprocess', None)
        self.df = self.prepare_df(csv_path, df_preprocess)
        self.num_classes = general_params['num_classes']
        self.use_labels = self.num_classes > 1
        if self.use_labels:
            assert 'label' in self.df.columns
        self.img_shape_yxc = general_params['img_shape_yxc']
        self.img_channels = self.img_shape_yxc[2]
        self.img_size_yx = self.img_shape_yxc[:2]
        img_preprocess = dataset_params.get('img_preprocess', None)
        assert img_preprocess in [None, 'random_crop_v1'], f'img_preprocess={img_preprocess} is not supported'
        self.img_preprocess = img_preprocess
        # TODO: support rectangular input
        assert self.img_size_yx[0] == self.img_size_yx[1], 'Rectangular images are not supported for now'
        self.horizontal_flip_p = dataset_params['horizontal_flip_p']
        self.vertical_flip_p = dataset_params['vertical_flip_p']
        # Set 25 M for full training
        self.train_mode = True
        self.full_training = True

    def __len__(self):
        return 25 * (10 ** 6) if self.full_training else len(self.df)

    def prepare_df(self, csv_path, df_preprocess):
        df = update_df_for_os(pd.read_csv(csv_path))
        if df_preprocess is not None:
            if df_preprocess == 'thr_512':
                src_size = len(df)
                df = df.loc[df['thr_512'] == True]
                upd_size = len(df)
            else:
                assert False, f'df_preprocess={df_preprocess} is not supported'
            print(f'Dataframe preprocessing ({df_preprocess}): src_size={src_size}, upd_size={upd_size}')
        return df

    def enable_mode(self, train_mode):
        print(f'Set dataset train_mode={train_mode}')
        self.train_mode = train_mode

    def train_preprocess(self, img):
        if self.img_preprocess is None:
            # Default preprocessing
            img = resize_img(img, self.img_size_yx)
        elif self.img_preprocess == 'random_crop_v1':
            img = random_crop_v1(img, self.img_size_yx)
        else:
            assert False
        return img

    def valid_preprocess(self, img):
        if self.img_preprocess is None:
            # Default preprocessing
            img = resize_img(img, self.img_size_yx)
        elif self.img_preprocess == 'random_crop_v1':
            img = center_crop_v2(img, self.img_size_yx)
        else:
            assert False
        return img

    def preprocess_img(self, img):
        src_img_size_yx = img.shape[:2]
        if src_img_size_yx != self.img_size_yx:
            if self.train_mode:
                img = self.train_preprocess(img)
            else:
                img = self.valid_preprocess(img)
        # BGR -> RGB, HWC -> CHW, disable cast to fp32 to reduce amount of data transfer to GPU
        img = np.ascontiguousarray(img[..., ::-1])
        img = np.transpose(img, (2, 0, 1))
        return img

    def augment_img(self, img):
        # img: CHW, UINT8
        if self.horizontal_flip_p > 0:
            img = np.ascontiguousarray(img[:, :, ::-1])
        if self.vertical_flip_p > 0:
            img = np.ascontiguousarray(img[:, ::-1, :])
        img = img.astype(np.uint8)
        return img

    def __getitem__(self, idx):
        if self.full_training:
            idx = idx % len(self.df)
        df_row = self.df.iloc[idx]
        img = load_image(df_row['path'], self.img_channels)
        img = self.preprocess_img(img)
        if self.train_mode:
            img = self.augment_img(img)
        label = df_row['cls_ind'] if self.use_labels else -1
        return img, label


# ----- Fast dataloader -----

class MultiEpochDataLoader(DataLoader):
    # Thanks to https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/4
    # Greatly speeds up training at least for Cifar datasets
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# ----- Trainer utils -----

def prepare_dataset(config):
    ds = BaseDataset(config)
    return ds


def prepare_dataloader(ds, config, rank=0, logger=None):
    dataloader_config = config['dataloader_params']
    DataLoaderInit = MultiEpochDataLoader if dataloader_config['use_fast_dataloader'] else DataLoader
    batch_size = get_total_batch_size(config=config)
    log_message(f'Dataloader batch size: {batch_size}', rank, logger)
    num_workers = get_num_workers(dataloader_config['num_workers'], rank, logger)
    data_loader = DataLoaderInit(ds,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=dataloader_config['pin_memory'],
                                 prefetch_factor=dataloader_config.get('prefetch_factor', None))
    return data_loader


def prepare_valid_dataloader(ds, dataloader_params, logger=None):
    data_loader = DataLoader(ds,
                             batch_size=dataloader_params['batch_size'],
                             shuffle=False,
                             drop_last=False,
                             num_workers=get_num_workers(dataloader_params.get('num_workers', 'AUTO'), logger),
                             pin_memory=dataloader_params.get('pin_memory', False),
                             prefetch_factor=dataloader_params.get('prefetch_factor', None))
    return data_loader
