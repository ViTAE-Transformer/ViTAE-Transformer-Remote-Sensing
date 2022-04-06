import os
import os.path as osp
import argparse
from PIL import Image
import cv2
from natsort import natsorted
import numpy as np

######### BGR!!!!!! #######
mask_mapping = {
    (0, 0, 0): 0,
    (63, 0, 0): 1,
    (63, 63, 0): 2,
    (0, 63, 0): 3,
    (127, 63, 0): 4,
    (191, 63, 0): 5,
    (255, 63, 0): 6,
    (63, 127, 0): 7,
    (127, 127, 0): 8,
    (127, 0, 0): 9,
    (191, 0, 0): 10,
    (255, 0, 0): 11,
    (127, 191, 0): 12,
    (191, 127, 0): 13,
    (255, 127, 0): 14,
    (155, 100, 0): 15
}

def get_args():
    parser = argparse.ArgumentParser(description='rgb2gray')
    parser.add_argument('--root_path', default='/public/data3/users/wangdi153/Dataset/isaid_patches/', help='root dir of iSAID dataset')
    parser.add_argument('--save_file_name', default='labels', help='save dir of converted gray masks of iSAID dataset')
    parser.add_argument('--phase', default='train', choices=['train', 'val', 'val_all'])

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    masks_path = osp.join(args.root_path, args.phase, 'masks')
    save_path = osp.join(args.root_path, args.phase, args.save_file_name)
    os.makedirs(save_path, exist_ok=True)
    mask_names = os.listdir(masks_path)
    mask_names = natsorted(mask_names)
    num_masks = len(mask_names)
    i = 0
    for token in mask_names:
        i = i+1
        if not token.endswith('.png'):
            continue
        save_name = osp.join(save_path, token)
        token_path = osp.join(masks_path, token)
        mask_array = cv2.imread(token_path)
        mask_gray = np.zeros(mask_array.shape[:2])
        for k, v in mask_mapping.items():
            mask_gray[(mask_array == k).all(axis=2)] = v
        assert mask_gray.max() <= 15, mask_gray.max()
        labels = np.unique(mask_gray)

        mask_gray = Image.fromarray(mask_gray.astype('uint8')).convert('L')
        mask_gray.save(save_name)
        print(f'Converted {token} to gray mask, [{i}/{num_masks}], {labels}')


if __name__ == '__main__':
    main()