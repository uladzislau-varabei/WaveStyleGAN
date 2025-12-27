import os
from glob import glob

import cv2
from tqdm import tqdm


def generate_video_v1(dir_pattern, video_fname, fps, freq_kimg=None, best_quality=True):
    # Note: read error message to check which h264 version is required.
    # Download from https://github.com/cisco/openh264/releases and extract in script's directory
    os.makedirs(os.path.dirname(video_fname), exist_ok=True)
    paths = sorted(glob(dir_pattern))
    img = cv2.imread(paths[0])
    h, w = img.shape[:2]
    if h > 1024 * 4 or w > 1024 * 4:
        best_quality = False
        print(f'Using best_quality=False since h={h}, w={w}')
    if best_quality:
        ext = 'mkv'
        fourcc_code = 'x264'
    else:
        ext = 'mp4'
        fourcc_code = 'mp4v'
    print(f'Video generation v1: ext={ext}, fourcc_code={fourcc_code}, freq_kimg={freq_kimg}, fps={fps}')
    kimg_postfix = f'_freq{freq_kimg}k' if freq_kimg is not None else ''
    video_fname = f'{video_fname}{kimg_postfix}_fps{fps}_v1.{ext}'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    video = cv2.VideoWriter(video_fname, fourcc, fps, (w, h))
    num_imgs = len(paths)
    for idx, path in tqdm(enumerate(paths), 'frames_v1', total=num_imgs):
        if freq_kimg is None:
            use_frame = True
        else:
            img_kimg = int(path.split('grid_')[1].split('.')[0])
            use_frame = img_kimg % int(freq_kimg * 1000) == 0
        if idx == (num_imgs - 1):
            use_frame = True
        if use_frame:
            video.write(cv2.imread(path))
    video.release()
    cv2.destroyAllWindows()
    print(f'Generated video: {video_fname}')


if __name__ == '__main__':
    dir_pattern = './imgs/*.png'
    video_fname = './videos/Model_video'

    freq_v = 2
    if freq_v == 1:
        freq_kimg = 50
        fps = 10
    elif freq_v == 2:
        freq_kimg = 10
        fps = 5
    else:
        assert False, f'freq_v={freq_v} is not supported'
    best_quality = True

    generate_video_v1(dir_pattern, video_fname, fps, freq_kimg=freq_kimg, best_quality=best_quality)
