import torch
import cv2
import os
import xlrd
import numpy as np
from torch.utils.data import Dataset


sep = os.sep


def get_filelist_frompath_tnscui(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.'+expname):
                id = file.split('.')[0]
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.'+expname):
                file_List.append(os.path.join(filepath, file))
    return file_List



# print(train_img_list)
# print(len(train_img_list))
# print(len(train_mask_list))


class TNSCUI_loader(Dataset):
    def __init__(self, datapath, image_size):
        """
        :param datapath: 图片所在文件夹路径
        :param image_size: 输入图片将会resize到的大小
        """

        # 读取测试集
        self.datapath = datapath
        mask_path = self.datapath.replace('image', 'mask')
        id_list = 'dataprocess/train_valid_test_id_test.xlsx'
        data_xlsx = xlrd.open_workbook(id_list)
        # 获取train集合
        table_train = data_xlsx.sheet_by_name('train')
        fold_id_train = [table_train.cell_value(i, 0) for i in range(1, table_train.nrows)]
        train_img_list = get_filelist_frompath_tnscui(datapath, 'bmp', sample_id=fold_id_train)
        train_img_list = np.array(train_img_list)
        np.random.shuffle(train_img_list)
        train_mask_list = np.array([mask_path + sep + i.split(sep)[-1] for i in train_img_list])
        # 获取vlaid集合
        table_valid = data_xlsx.sheet_by_name('valid')
        fold_id_valid = [table_valid.cell_value(i, 0) for i in range(1, table_valid.nrows)]
        valid_img_list = get_filelist_frompath_tnscui(datapath, 'bmp', sample_id=fold_id_valid)
        valid_img_list = np.array(valid_img_list)
        np.random.shuffle(valid_img_list)
        valid_mask_list = np.array([mask_path + sep + i.split(sep)[-1] for i in valid_img_list])

        # 拼接train和valid
        train_img_list = np.hstack((train_img_list, valid_img_list))
        train_mask_list = np.hstack((train_mask_list, valid_mask_list))


        self.train_img_list = train_img_list
        self.train_mask_list = train_mask_list
        self.image_size = image_size

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.train_img_list[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'mask')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # 统一图片大小
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size))
        # 还原成3维
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        return image_path, image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.train_img_list)

    def get_img_list(self):
        return self.train_img_list

    def get_gt_list(self):
        return self.train_mask_list


if __name__ == "__main__":
    datapath = 'G:/graduation project/TN-SCUI2020/my_code/1_or_data/image'
    tnscui_dataset = TNSCUI_loader(datapath, image_size=512)
    print("数据个数：", len(tnscui_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=tnscui_dataset,
                                               batch_size=50,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
