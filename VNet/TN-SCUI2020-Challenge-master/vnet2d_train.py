import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule, AGVnet2dModule
import numpy as np
import pandas as pd
import xlrd

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


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csvdata = pd.read_csv('dataprocess\segandclassifydata_zch.csv')
    # trainData = csvdata.iloc[:, :].values
    # print(type(trainData))
    # np.random.shuffle(trainData)
    # print(type(trainData))
    # labeldata = trainData[:, 0]
    # imagedata = trainData[:, 1]
    # maskdata = trainData[:, 2]
    # print(imagedata)
    # print(maskdata)

    # 读取测试集
    img_path = 'G:/graduation project/TN-SCUI2020/my_code/1_or_data/image'
    mask_path = 'G:/graduation project/TN-SCUI2020/my_code/1_or_data/mask'
    id_list = 'dataprocess/train_valid_test_id.xlsx'
    data_xlsx = xlrd.open_workbook(id_list)
    # 获取train集合
    table_train = data_xlsx.sheet_by_name('train')
    fold_id_train = [table_train.cell_value(i, 0) for i in range(1, table_train.nrows)]
    train_img_list = get_filelist_frompath_tnscui(img_path, 'bmp', sample_id=fold_id_train)
    train_img_list = np.array(train_img_list)
    np.random.shuffle(train_img_list)
    train_mask_list = np.array([mask_path + sep + i.split(sep)[-1] for i in train_img_list])
    # 获取vlaid集合
    table_valid = data_xlsx.sheet_by_name('valid')
    fold_id_valid = [table_train.cell_value(i, 0) for i in range(1, table_valid.nrows)]
    valid_img_list = get_filelist_frompath_tnscui(img_path, 'bmp', sample_id=fold_id_valid)
    valid_img_list = np.array(valid_img_list)
    np.random.shuffle(valid_img_list)
    valid_mask_list = np.array([mask_path + sep + i.split(sep)[-1] for i in valid_img_list])

    # 拼接train和valid
    train_img_list = np.hstack((train_img_list, valid_img_list))
    train_mask_list = np.hstack((train_mask_list, valid_mask_list))

    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient")
    Vnet2d.train(train_img_list, train_mask_list, "Vnet2d.pd", "log\\segmeation\\vnet2dtest\\", 0.001, 0.5, 10, 3, restore=False, step=6000)


def trainag():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\segandclassifydata.csv')
    trainData = csvdata.iloc[:, :].values
    np.random.shuffle(trainData)
    labeldata = trainData[:, 0]
    imagedata = trainData[:, 1]
    maskdata = trainData[:, 2]

    agVnet2d = AGVnet2dModule(512, 512, channels=1, costname="dice coefficient")
    agVnet2d.train(imagedata, maskdata, "agVnet2d.pd", "log\\segmeation\\agvnet2d\\", 0.001, 0.5, 10, 1)


if __name__ == '__main__':
    train()
    print('success')
