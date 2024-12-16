import numpy as np
from PIL import Image
import os

# 设置包含.npy文件的目录路径
# directory = 'path_to_your_npy_directory'
directory = r'E:\UltraLight-VM-UNet-main\data\dataset_isic17'


# 遍历目录中的所有.npy文件
for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        # 构建.npy文件的完整路径
        npy_file_path = os.path.join(directory, filename)

        # 步骤1: 读取.npy文件
        # 假设.npy文件中存储的是一个包含多个图像的数组
        images_array = np.load(npy_file_path)

        # 步骤2: 遍历数组中的每个图像
        for i, image_array in enumerate(images_array):
            # 步骤2.1: 数据转换（如果需要）
            # 确保数据在0-255范围内，并且转换为整数类型
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

            # 步骤3: 将图像数组转换为PIL图像
            img = Image.fromarray(image_array)

            # 步骤4: 保存为.jpg格式
            # 构建输出图像的文件名，包括序号以区分同一.npy文件中的不同图像
            output_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            img.save(os.path.join(directory, output_filename), 'JPEG')

print("所有图像已从.npy文件转换并保存为.jpg格式。")