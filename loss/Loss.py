import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from models.discriminator import Discriminator_STE
from PIL import Image
import numpy as np


# 定义了一个计算Gram矩阵的函数。Gram矩阵用于衡量特征图之间的相关性
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    # 获取输入特征图的尺寸信息
    # 这些变量分别表示批次大小（batch size）、通道数（channels）、高度（height）和宽度（width）
    (b, ch, h, w) = feat.size()
    # 将特征图的维度进行变换，将每个像素点的特征向量展平
    # 具体来说，它将特征图的形状从 (b, ch, h, w) 变为 (b, ch, h * w)
    feat = feat.view(b, ch, h * w)
    # ???将 feat 的维度进行转置，将第1维和第2维进行交换 这样做是为了后续计算格拉姆矩阵做准备
    feat_t = feat.transpose(1, 2)
    # ???计算格拉姆矩阵
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    # 返回计算得到的格拉姆矩阵
    return gram


def visual(image):
    # 对输入的图像 image 进行维度变换和数据类型转换
    # ???首先，将图像的第1维和第2维进行交换，然后将第2维和第3维进行交换,这样做是为了将通道维度放在最后
    # 并将图像的维度从 (channels, height, width) 变为 (height, width, channels)
    # 接着，使用 detach() 方法将图像从计算图中分离
    # 然后使用 cpu() 方法将图像从GPU内存中移动到CPU内存
    # 最后，使用 numpy() 方法将图像转换为NumPy数组，并将其赋值给变量 im
    im = image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    # 将NumPy数组 im[0] 转换为PIL图像对象，并调用 show() 方法显示图像
    Image.fromarray(im[0].astype(np.uint8)).show()


# 计算Dice损失
def dice_loss(input, target):
    # 对输入 input 进行sigmoid激活，将其限制在0到1之间
    input = torch.sigmoid(input)

    # 将输入 input 和目标 target 展平为二维张量
    # contiguous() 方法用于确保张量在内存中是连续存储的，以便后续的视图变换操作能够正确执行
    # view() 方法用于改变张量的形状，这里将其变为 (batch_size, -1) 的形状,其中 -1 表示自动计算该维度的大小
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    # ???
    input = input
    target = target

    # 计算Dice系数和Dice损失
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    # 使用 torch.mean 函数计算Dice系数的平均值，并将其赋值给变量 dice_loss
    dice_loss = torch.mean(d)
    return 1 - dice_loss


class LossWithGAN_STE(nn.Module):
    # logPath是日志文件的路径，extractor是特征提取器模型，Lamda是一个参数，lr是学习率，betasInit是Adam优化器的beta参数的初始值
    def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        # 创建了一个L1损失函数
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        # 判别器模型
        self.discriminator = Discriminator_STE(3)  ## local_global sn patch gan
        # 一个Adam优化器
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.lamda = Lamda
        # 记录训练过程中的日志的SummaryWriter对象
        self.writer = SummaryWriter(logPath)

    # input是输入数据，mask是遮挡掩码，x_o1、x_o2、x_o3是一些中间输出 output是网络模型的输出
    # mm是一个额外的遮挡掩码，gt是真实标签，count是计数器，epoch是当前的训练轮数
    def forward(self, input, mask, x_o1, x_o2, x_o3, output, mm, gt, count, epoch):
        # 将判别器的梯度清零，以便进行反向传播
        self.discriminator.zero_grad()
        # 使用判别器对真实标签和遮挡掩码进行预测，并计算预测结果的平均值
        # 然后将其乘以 - 1，以得到真实标签的损失
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        # 使用判别器对网络模型的输出和遮挡掩码进行预测，并计算预测结果的平均值
        # 然后将其乘以1，以得到网络模型输出的损失
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        # 计算判别器的损失，使用了SN-patch-GAN损失函数
        D_loss = torch.mean(F.relu(1. + D_real)) + torch.mean(F.relu(1. + D_fake))  # SN-patch-GAN loss
        # 计算判别器对网络模型输出的损失
        D_fake = -torch.mean(D_fake)  # SN-Patch-GAN loss

        # 对判别器的参数进行反向传播和优化
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()

        # 判别器的损失写入日志
        self.writer.add_scalar('LossD/Discrinimator loss', D_loss.item(), count)

        # 生成一个修复后的输出，通过将遮挡区域的输入与网络模型的输出进行组合
        output_comp = mask * input + (1 - mask) * output
        # import pdb;pdb.set_trace()
        # 计算修复后输出的遮挡区域损失和有效区域损失
        holeLoss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = 2 * self.l1(mask * output, mask * gt)

        # 计算遮挡掩码的损失
        mask_loss = dice_loss(mm, 1 - mask)
        ### MSR loss ###
        # 对遮挡掩码和真实标签进行插值，得到不同尺度下的遮挡掩码和真实标签
        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        # 计算多尺度重建损失
        msrloss = 8 * self.l1((1 - mask) * x_o3, (1 - mask) * gt) + 0.8 * self.l1(mask * x_o3, mask * gt) + \
                  6 * self.l1((1 - masks_b) * x_o2, (1 - masks_b) * imgs2) + 1 * self.l1(masks_b * x_o2,
                                                                                         masks_b * imgs2) + \
                  5 * self.l1((1 - masks_a) * x_o1, (1 - masks_a) * imgs1) + 0.8 * self.l1(masks_a * x_o1,
                                                                                           masks_a * imgs1)
        # 使用特征提取器提取修复后输出、网络模型输出和真实标签的特征
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        # 计算感知损失，使用了L1损失函数
        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        # 计算风格损失，使用了格拉姆矩阵和L1损失函数
        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                       gram_matrix(feat_gt[i]))
        # 修复损失、有效区域损失、多尺度重建损失、感知损失和风格损失写入日志
        """ if self.numOfGPUs > 1:
            holeLoss = holeLoss.sum() / self.numOfGPUs
            validAreaLoss = validAreaLoss.sum() / self.numOfGPUs
            prcLoss = prcLoss.sum() / self.numOfGPUs
            styleLoss = styleLoss.sum() / self.numOfGPUs """
        self.writer.add_scalar('LossG/Hole loss', holeLoss.item(), count)
        self.writer.add_scalar('LossG/Valid loss', validAreaLoss.item(), count)
        self.writer.add_scalar('LossG/msr loss', msrloss.item(), count)
        self.writer.add_scalar('LossPrc/Perceptual loss', prcLoss.item(), count)
        self.writer.add_scalar('LossStyle/style loss', styleLoss.item(), count)
        # 计算并返回生成器的综合损失
        GLoss = msrloss + holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake + 1 * mask_loss
        self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()
