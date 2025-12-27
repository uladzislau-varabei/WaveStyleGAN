import os
from glob import glob

import cv2
import pandas as pd
from tqdm import tqdm


def check_is_image(p):
    ext = p.rsplit('.', 1)[1].lower()
    return ext in ['png', 'jpg', 'jpeg']


def check_size(img_size_yx, thr):
    return img_size_yx[0] > thr and img_size_yx[1] > thr


def prepare_FFHQ_dataset(data_dir, dst_csv_path):
    imgs_paths = sorted([dir for dir in glob(os.path.join(data_dir, '*', '*.png'))])
    df = pd.DataFrame.from_dict({'path': imgs_paths})
    os.makedirs(os.path.dirname(dst_csv_path), exist_ok=True)
    df.to_csv(dst_csv_path, index=False)
    print(f'Saved FFHQ csv to {dst_csv_path}')


def prepare_fish_dataset(data_dir, dst_csv_path):
    cls_dirs = sorted(glob(os.path.join(data_dir, '*')))
    imgs_paths = []
    cls_inds = []
    cls_labels = []
    for idx, cls_dir in enumerate(cls_dirs):
        cls_paths = sorted([p for p in glob(os.path.join(cls_dir, '*')) if check_is_image(p)])
        imgs_paths += cls_paths
        cls_inds += len(cls_paths) * [idx]
        cls_label = os.path.split(cls_dir)[1].replace(' ', '_')
        cls_labels += len(cls_paths) * [cls_label]
    df = pd.DataFrame({'path': imgs_paths, 'cls_ind': cls_inds, 'cls_label': cls_labels})
    print('Fish dataset cls samples:')
    print(df['cls_label'].value_counts())
    n_labels = len(df['cls_label'].unique())
    print(f'Total_size={len(df)}, n_labels={n_labels}')
    os.makedirs(os.path.dirname(dst_csv_path), exist_ok=True)
    df.to_csv(dst_csv_path, index=False)
    print(f'Saved Fish csv to {dst_csv_path}')


def prepare_landscape_dataset(data_dir, dst_csv_path, collect_stats=True):
    paths = sorted(glob(os.path.join(data_dir, '*')))
    imgs_paths = [p for p in paths if check_is_image(p)]
    df = pd.DataFrame({'path': imgs_paths})
    num_imgs = len(df)
    if collect_stats:
        shapes = []
        thr_res_256 = []
        thr_res_384 = []
        thr_res_512 = []
        thr_res_768 = []
        thr_res_1024 = []
        for p in tqdm(imgs_paths, 'Img'):
            img = cv2.imread(p)
            shape = img.shape
            shapes.append(list(shape))
            thr_res_256.append(check_size(shape, 256))
            thr_res_384.append(check_size(shape, 384))
            thr_res_512.append(check_size(shape, 512))
            thr_res_768.append(check_size(shape, 768))
            thr_res_1024.append(check_size(shape, 1024))
        columns = ['img_shape_yxc', 'thr_256', 'thr_384', 'thr_512', 'thr_768', 'thr_1024']
        values = [shapes, thr_res_256, thr_res_384, thr_res_512, thr_res_768, thr_res_1024]
        print(f'Landscape dataset (size={num_imgs}) stats:')
        for column_name, column_values in zip(columns, values):
            df[column_name] = column_values
            print(f'{column_name}:\n{df[column_name].value_counts()}')
    df.to_csv(dst_csv_path, index=False)
    print(f'Saved Landscape dataset (size={num_imgs}) csv to {dst_csv_path}')


if __name__ == '__main__':
    data_dir = '.../FFHQ/'
    dst_csv_path = os.path.join('.', 'dataset_csvs', 'FFHQ_v1.csv')
    # prepare_FFHQ_dataset(data_dir, dst_csv_path)

