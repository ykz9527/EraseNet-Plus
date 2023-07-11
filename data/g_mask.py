import os
import cv2
import numpy as np

# 输入文件夹路径
image_folder = '/root/autodl-tmp/dataset/syn_train/img'
gt_folder = '/root/autodl-tmp/dataset/syn_train/label'

# 输出文件夹路径
output_mask_folder = '/root/autodl-tmp/dataset/syn_train/mask'

# 获取输入文件夹中的图像文件列表
image_files = os.listdir(image_folder)

# 遍历每个图像文件
for image_file in image_files:
    # 构建图像文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    # 构建对应的gt文件的完整路径
    gt_file = image_file
    gt_path = os.path.join(gt_folder, gt_file)

    # 读取图像文件
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 读取gt文件
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # 计算灰度差值
    diff = cv2.absdiff(image, gt)

    # 设置阈值，将大于阈值的像素设为255，小于等于阈值的像素设为0
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白的mask图像
    mask = np.zeros_like(image)

    # 绘制轮廓到mask图像上
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 构建输出文件的完整路径
    output_mask_path = os.path.join(output_mask_folder, gt_file)

    # 保存mask图像
    cv2.imwrite(output_mask_path, mask)