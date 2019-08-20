import os
import glob
import math
import cv2
import numpy as np
work_place = '/home/scau638/awork/'
image_format = ['jpg', 'jpeg', 'JPG', 'png']


def get_data_files(path):
    inputs = []
    labels = []
    for ext in image_format:
        inputs.extend(glob.glob(
            os.path.join(path + 'input/', '*.{}'.format(ext))))
    for f in inputs:
        num, end = f.split('/')[-1].split('.')
        labels.append(path + 'label/' + num.split('_')[0] + '.' + end)

    return inputs, labels


def computer_psn(im1, im2, m=255.0, bgr=True):
    if not bgr:
        im1 = im1[:, :, ::-1]
        im2 = im2[:, :, ::-1]
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2GRAY)
    im_dff = gray2 - gray1
    # imdff = imdff.astype(np.float32)
    mse = np.sqrt(np.mean(im_dff * im_dff))
    return 20 * math.log10(m / mse)


def cpsnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)


def psnr2(img1, img2):
    # mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    # res = get_data_files(work_place + 'dataset/derain/fu/')
    # for i in range(len(res[0])):
    #     print(res[0][i], res[1][i])
    pass
