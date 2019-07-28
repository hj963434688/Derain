import cv2
import numpy as np
import math
from skimage.measure import compare_ssim
import compu_ssim as cs


def fun(im1, im2, m = 255.0, BGR=True):
    if not BGR:
        im1 = im1[:, :, ::-1]
        im2 = im2[:, :, ::-1]
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2GRAY)
    imdff = gray2 - gray1
    # imdff = imdff.astype(np.float32)
    rmse = np.sqrt(np.mean(imdff * imdff))
    return 20 * math.log10(m / rmse)


def funssim(im1, im2, m = 255.0, BGR=False):
    if not BGR:
        im1 = im1[:, :, ::-1]
        im2 = im2[:, :, ::-1]
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2GRAY)

    score, diff = compare_ssim(gray1, gray2, full=True)

    return score


def funssimx(im1, im2, m = 255.0, BGR=True):
    if not BGR:
        im1 = im1[:, :, ::-1]
        im2 = im2[:, :, ::-1]
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2GRAY)

    score= cs.compute_ssim(gray1, gray2)

    return score


def psnr1(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)


def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# def ssim1(img1, img2):

if __name__ == '__main__':
    print(math.log(4, 2))
