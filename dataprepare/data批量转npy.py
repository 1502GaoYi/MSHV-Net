import numpy as np
from PIL import Image
import os


for i in ['dataset_isic17','dataset_isic18']:
    for j in ['train', 'val']:
        train = "train"
        val = "test"
        for k in ['images','masks']:
            jpg_directory = 'data/'+i+'/'+j+'/'+k
            print(jpg_directory)

            relative_path = jpg_directory
            print(relative_path)
            absolute_path = r'E:/MSHV-Net/{}'.format(jpg_directory)    #根据实际情况转换绝对路径
            print(absolute_path,'这是绝对路径')

            jpg_directory = absolute_path
            print(jpg_directory,'jpg_directory这是绝对路径')

            # 读取目录中的所有.jpg文件，并将其转换为NumPy数组
            images = []
            for filename in os.listdir(jpg_directory):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    jpg_file_path = os.path.join(jpg_directory, filename)
                    img = Image.open(jpg_file_path)
                    img_array = np.array(img)
                    images.append(img_array)

            # 将列表中的所有图片数组合并成一个多维数组
            # 假设所有图片尺寸相同，合并成(N, H, W, C)形状的数组，其中
            # N 是图片数量，H 是高度，W 是宽度，C 是通道数（对于RGB图片是3）
            images_array = np.stack(images)

            # 保存合并后的数组到.npy文件
            # np.save('train/images2/all_images.npy', images_array)
            # np.save(r'E:\UltraLight-VM-UNet-main\data\dataset_isic17\mask_val.npy', images_array)
            if j == 'train':
                if k == 'images':    #结合情况调整绝对路径
                    np.save('E:/MSHV-Net/data/{}'.format(i)+'/'+'data_'+'train.npy',images_array)
                    print(jpg_directory+'/'+'data_'+'train.npy','------------------------------------------------------------------------')
                if k == 'masks':
                    np.save('E:/MSHV-Net/data/{}'.format(i)+'/'+'mask_'+'train.npy',images_array)

            if j == 'val':
                # val  test
                if k == 'images':
                    np.save('E:/MSHV-Net/data/{}'.format(i) + '/' + 'data_' + 'val.npy',images_array)
                    np.save('E:/MSHV-Net/data/{}'.format(i) + '/' + 'data_' + 'test.npy',images_array)
                if k == 'masks':
                    np.save('E:/MSHV-Net/data/{}'.format(i) + '/' + 'mask_' + 'val.npy',images_array)
                    np.save('E:/MSHV-Net/data/{}'.format(i) + '/' + 'mask_' + 'test.npy',images_array)
            print('{}转换成功'.format(jpg_directory))


