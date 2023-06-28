import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import utils
from data.dataloader import ErasingData
from loss.Loss import LossWithGAN_STE
from models.Model import VGG16FeatureExtractor
from models.sa_gan import STRnet2

torch.set_num_threads(5)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ### set the gpu as No....

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=1,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='~/tmp',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='/home/yukai/data/train/all_images')
parser.add_argument('--pretrained', type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=500, help='epochs')
args = parser.parse_args()


# 用于可视化图像
def visual(image):
    # 将输入的图像进行维度转换，并将其从Tensor类型转换为NumPy数组
    im = image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    # 将NumPy数组转换为图像，并显示出来
    Image.fromarray(im[0].astype(np.uint8)).show()


# 判断是否有可用的CUDA加速设备，并打印相关信息
cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    # 启用cudnn加速
    cudnn.enable = True
    # 设置cudnn加速的模式为自动寻找最适合当前硬件的配置
    cudnn.benchmark = True

batchSize = args.batchSize
# 设置加载图像的尺寸
loadSize = (args.loadSize, args.loadSize)

# 检查保存模型的路径是否存在，如果不存在则创建
if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

# 设置数据集的根目录
dataRoot = args.dataRoot

# import pdb;pdb.set_trace()
# 创建一个数据集对象
Erase_data = ErasingData(dataRoot, loadSize, training=True)
# 创建一个数据加载器，用于批量加载数据
Erase_data = DataLoader(Erase_data, batch_size=batchSize,
                        shuffle=True, num_workers=args.numOfWorkers, drop_last=False, pin_memory=True)

# 创建一个名为netG的模型对象
netG = STRnet2(3)

# 判断是否有预训练模型的路径,有则加载
if args.pretrained != '':
    print('loaded ')
    netG.load_state_dict(torch.load(args.pretrained))

# 获取可用的GPU数量
numOfGPUs = torch.cuda.device_count()

# 判断是否有可用的CUDA加速设备,有则将模型移动到CUDA设备上
if cuda:
    netG = netG.cuda()
    if numOfGPUs > 1:
        # 使用DataParallel将模型包装起来，以实现多GPU并行计算
        netG = nn.DataParallel(netG, device_ids=range(numOfGPUs))

# 初始化计数器
count = 1

# 创建一个Adam优化器，用于优化netG模型的参数
G_optimizer = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))

# 一个损失函数对象
criterion = LossWithGAN_STE(args.logPath, VGG16FeatureExtractor(), lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)

if cuda:
    # 损失函数移动到CUDA设备上
    criterion = criterion.cuda()

    if numOfGPUs > 1:
        criterion = nn.DataParallel(criterion, device_ids=range(numOfGPUs))

print('OK!')
# 设置训练的总轮数
num_epochs = args.num_epochs
# 设置PyTorch自动求导时检测异常的模式为True
# torch.autograd.set_detect_anomaly(True)
for i in range(1, num_epochs + 1):
    # 将模型设置为训练模式
    netG.train()

    # 遍历数据加载器中的每个批次
    for k, (imgs, gt, masks, path) in enumerate(Erase_data):
        if cuda:
            # 将输入图像移动到CUDA设备上
            imgs = imgs.cuda()
            # 将真实图像移动到CUDA设备上
            gt = gt.cuda()
            # 将遮挡图像移动到CUDA设备上
            masks = masks.cuda()
        # 清空模型的梯度
        netG.zero_grad()

        # 使用netG模型生成合成图像
        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        # 计算生成器的损失
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, count, i)
        # 对损失进行求和
        G_loss = G_loss.sum()
        # 清空生成器优化器的梯度
        G_optimizer.zero_grad()
        # 反向传播计算梯度
        # https://blog.csdn.net/MilanKunderaer/article/details/121425885
        #G_loss.backward()
        loss1 = G_loss.detach_().requires_grad_(True)
        loss1.backward()
        # 更新生成器的参数
        G_optimizer.step()

        # 打印生成器的损失
        print('[{}/{}] Generator Loss of epoch{} is {}'.format(k, len(Erase_data), i, loss1.item()))

        # 更新计数器
        count += 1

    # 保存模型的参数
    if i % 10 == 0:
        if numOfGPUs > 1:
            torch.save(netG.module.state_dict(), args.modelsSavePath +
                       '/STE_{}.pth'.format(i))
        else:
            torch.save(netG.state_dict(), args.modelsSavePath +
                       '/STE_{}.pth'.format(i))
