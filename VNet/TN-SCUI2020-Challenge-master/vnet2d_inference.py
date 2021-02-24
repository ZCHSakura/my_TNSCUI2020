import os

import xlrd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule, AGVnet2dModule
# from dataprocess.utils import calcu_iou
import cv2
import os
import numpy as np

def getIOU(SR,GT):
    """
    都是二值图
    :param SR:
    :param GT:
    :return:
    """
    TP = (SR+GT == 2).astype(np.float32)
    FP = (SR+(1-GT) == 2).astype(np.float32)
    FN = ((1-SR)+GT == 2).astype(np.float32)

    IOU = float(np.sum(TP))/(float(np.sum(TP+FP+FN)) + 1e-6)

    return IOU


def predict_test():
    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                          model_path="log/segmeation/vnet2d6000lr-3-balance/model/Vnet2d.pd")
    test_image_path = os.path.abspath('../..') + r"\Vnet\test\image"
    test_gt_path = os.path.abspath('../..') + r"\1_or_data\mask"
    test_mask_path = os.path.abspath('../..') + r"\Vnet\test\mask"
    print(test_gt_path)
    allimagefiles = os.listdir(test_image_path)
    # counter = 0
    all_iou = []
    for imagefile in allimagefiles:
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        gtfilepath = os.path.join(test_gt_path, imagefile)
        gt_image = cv2.imread(gtfilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        # 计算IOU，先将GT和预测图像化为二值图
        gt_image_iou = (gt_image / 255.).astype(np.float)
        new_mask_image_iou = ((new_mask_image / 255.) > 0.8).astype(np.float)
        IOU = getIOU(gt_image_iou, new_mask_image_iou)
        print(IOU)
        all_iou.append(IOU)
        if IOU < 0.7:
            print(imagefile)
        # counter += 1
        # if counter == 50:
        #     print('我该跳出了')
        #     break

        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)

    mean_iou = np.mean(all_iou)
    print('Mean_iou:', mean_iou)


def predict_testzch():
    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                          model_path="log/segmeation/vnet2d_author/model/Vnet2d.pd")
    test_image_path = os.path.abspath('../..') + r"\1_or_data\image"
    test_gt_path = os.path.abspath('../..') + r"\1_or_data\mask"
    test_mask_path = os.path.abspath('../..') + r"\Vnet\test\mask"
    id_list = 'dataprocess/train_valid_test_id.xlsx'
    data_xlsx = xlrd.open_workbook(id_list)
    # 获取test集合
    table_test = data_xlsx.sheet_by_name('test')
    fold_id_test = [table_test.cell_value(i, 0) + '.bmp' for i in range(1, table_test.nrows)]
    print(fold_id_test)
    print(len(fold_id_test))
    # counter = 0
    all_iou = []
    for imagefile in fold_id_test:
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        gtfilepath = os.path.join(test_gt_path, imagefile)
        gt_image = cv2.imread(gtfilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        # 计算IOU，先将GT和预测图像化为二值图
        gt_image_iou = (gt_image / 255.).astype(np.float)
        new_mask_image_iou = ((new_mask_image / 255.) > 0.8).astype(np.float)
        IOU = getIOU(gt_image_iou, new_mask_image_iou)
        print(IOU)
        all_iou.append(IOU)
        if IOU < 0.7:
            print(imagefile)
        # counter += 1
        # if counter == 50:
        #     print('我该跳出了')
        #     break

        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)

    mean_iou = np.mean(all_iou)
    print('Mean_iou:', mean_iou)


def predict_testag():
    Vnet2d = AGVnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                            model_path="log\segmeation\\agvnet2d\model\\agVnet2d.pd")
    test_image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\image"
    test_mask_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\agvnet_mask"
    allimagefiles = os.listdir(test_image_path)
    for imagefile in allimagefiles:
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)


if __name__ == "__main__":
    predict_testzch()
    # predict_testag()
    print('success')
