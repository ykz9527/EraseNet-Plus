import os
import math
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataloader import devdata
from scipy import signal, ndimage
import gauss
import logging

# ssim 0.8 psnr 30左右
parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='/root/autodl-tmp/test/results/sn_tv/WithMaskOutput/',
                    help='results')
parser.add_argument('--gt_path', type=str, default='/root/autodl-tmp/test/all_labels',
                    help='labels')
parser.add_argument('--logFile',default='/root/autodl-tmp/test/test.out', type=str, help='Path to log file')
args = parser.parse_args()
# 配置日志记录
logging.basicConfig(filename=args.logFile, level=logging.INFO, format='%(message)s')

sum_psnr = 0
sum_ssim = 0
sum_AGE = 0 
sum_pCEPS = 0
sum_pEPS = 0
sum_mse = 0

count = 0
sum_time = 0.0
l1_loss = 0

img_path = args.target_path
gt_path = args.gt_path

# 计算结构相似性(SSIM)指数的函数
def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
  #  import pdb;pdb.set_trace()
    mu1 = signal.fftconvolve(img1, window, mode = 'valid')
    mu2 = signal.fftconvolve(img2, window, mode = 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode = 'valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode = 'valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode = 'valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))


# 计算多尺度结构相似性(MSSSIM)指数的函数
def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map = True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(img1, downsample_filter, 
                                                mode = 'reflect')
        filtered_im2 = ndimage.filters.convolve(img2, downsample_filter, 
                                                mode = 'reflect')
        im1 = filtered_im1[: : 2, : : 2]
        im2 = filtered_im2[: : 2, : : 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0 : level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0 : level - 1]), weight[0 : level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power


def ImageTransform(loadSize, cropSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
      #  RandomCrop(size=cropSize),
        #RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])

def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


imgData = devdata(dataRoot=img_path, gtRoot=gt_path)
data_loader = DataLoader(imgData, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
print(len(data_loader))
# 这部分代码遍历数据加载器中的每个批次，计算输入图像和gt之间的均方误差(MSE)，并累加到sum_mse中。
# path是图像的路径信息，count用于计数。
for k, (img,lbl,path) in enumerate(data_loader):
	##import pdb;pdb.set_trace()
	mse = ((lbl - img)**2).mean()
	sum_mse += mse
	print(path,count, 'mse: ', mse)
    # 这部分代码检查MSE是否为0，如果为0，则跳过当前批次的处理。
    # 否则，增加计数count，计算PSNR（峰值信噪比）并将其累加到sum_psnr中。PSNR的计算公式为10 * log10(1 / MSE)。
	if mse == 0:
		continue
	count += 1
	psnr = 10 * math.log10(1/mse)
	sum_psnr += psnr
	print(path,count, ' psnr: ', psnr)
	#l1_loss += nn.L1Loss()(img, lbl)

    # 这部分代码提取了输入图像和gt图像的亮度信息，并计算了亮度之间的差异（即Diff）。根据差异值计算了平均绝对差异(AGE)。
	R = lbl[0,0,:, :]
	G = lbl[0,1,:, :]
	B = lbl[0,2,:, :]

	YGT = .299 * R + .587 * G + .114 * B

	R = img[0,0,:, :]
	G = img[0,1,:, :]
	B = img[0,2,:, :]

	YBC = .299 * R + .587 * G + .114 * B
	Diff = abs(np.array(YBC*255) - np.array(YGT*255)).round().astype(np.uint8)
	AGE = np.mean(Diff)
	print(' AGE: ', AGE)
    # 这部分代码使用前面定义的msssim函数，计算输入图像和gt图像的MSSSIM指数，并将其累加到sum_ssim中。
	mssim = msssim(np.array(YGT*255), np.array(YBC*255))
	sum_ssim += mssim
	print(count, ' ssim:', mssim)
    # 这部分代码根据设定的阈值，将差异值大于阈值的像素标记为错误(Errors)。
    # 计算错误像素点的数量(EPs)和比例(pEPs)，并将pEPs累加到sum_pEPS中。
	threshold = 20
	Errors = Diff > threshold
	EPs = sum(sum(Errors)).astype(float)
	pEPs = EPs / float(512*512)
	print(' pEPS: ' , pEPs)
	sum_pEPS += pEPs
    # 这部分代码计算连通错误像素的数量(CEPs)和比例(pCEPs)。
    # 使用二值形态学的腐蚀操作(ndimage.binary_erosion)对Errors进行处理，以提取连通的错误像素。
    # structure定义了腐蚀操作的结构元素。CEPs和pCEPs分别是连通错误像素的数量和比例，将它们累加到sum_pCEPS中。
	########################## CEPs and pCEPs ################################
	structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
	sum_AGE+=AGE
	erodedErrors = ndimage.binary_erosion(Errors, structure).astype(Errors.dtype)
	CEPs = sum(sum(erodedErrors))
	pCEPs = CEPs / float(512*512)
	print(' pCEPS: ' , pCEPs)
	sum_pCEPS += pCEPs

logging.info(f"psnr: {sum_psnr}")
logging.info(f"average mse: {sum_mse / count}")
logging.info(f"average psnr: {sum_psnr / count}")
logging.info(f"average ssim: {sum_ssim / count}")
logging.info(f"average AGE: {sum_AGE / count}")
logging.info(f"average pEPS: {sum_pEPS / count}")
logging.info(f"average pCEPS: {sum_pCEPS / count}")