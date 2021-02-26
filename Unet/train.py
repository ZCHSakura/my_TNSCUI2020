import argparse
import datetime

from utils import *
from model.unet_model import UNet
from dataprocess.dataset import TNSCUI_loader
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
import torch
import os
import cv2


def train(net, device, config):
    myprint(config.record_file, '-----------------------%s-----------------------------' % config.Task_name)
    # 加载训练集
    tnscui_dataset = TNSCUI_loader(config.data_path, image_size=config.image_size)
    train_loader = torch.utils.data.DataLoader(dataset=tnscui_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    train_len = len(train_loader)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=config.lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    writer = SummaryWriter(log_dir=config.log_dir)
    Iter = 0
    last_epoch = 0
    if config.last_epoch != 0:
        last_epoch = config.last_epoch
        Iter = last_epoch * len(tnscui_dataset) / config.batch_size
        config.model_path = os.path.join(config.result_path, 'models')
        checkpoint = torch.load(os.path.join(config.model_path, 'epoch%s.pkl' % last_epoch), map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        myprint(config.record_file, 'reload success! from epoch %s' % last_epoch)

    # 训练epochs次
    for epoch in range(last_epoch, config.num_epochs):
        tic = datetime.datetime.now()

        # 训练模式
        net.train()
        epoch_loss = 0
        length = 0

        # 为后面的测试做准备
        image_paths = []
        # 按照batch_size开始训练
        for i, sample in enumerate(train_loader):
            (image_path, image, label) = sample
            image_paths.append(image_path)
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            # print('Loss/train', loss.item())
            # # 保存loss值最小的网络参数
            # if loss < best_loss:
            #     best_loss = loss
            #     print('best in %s epoch' % epoch)
            #     torch.save(net.state_dict(), os.path.join(config.model_path, 'best_unet_score.pkl'))
            # 更新参数
            epoch_loss += float(loss)
            loss.backward()
            optimizer.step()

            length += 1
            Iter += 1
            writer.add_scalars('Loss', {'loss': loss}, Iter)

            print_content = 'batch_loss:' + str(loss.data.cpu().numpy()) + '\n'
            printProgressBar(i+1, train_len, content=print_content)

        epoch_loss = epoch_loss / length
        myprint(config.record_file, 'Epoch [%d/%d], Loss: %.4f' % (epoch + 1, config.num_epochs, epoch_loss))

        # 计时结束
        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
        myprint(config.record_file, time_str)

        # 保存模型
        if (epoch + 1) % config.save_model_step == 0:
            save_unet = net.state_dict()
            checkpoint = {
                'state_dict': save_unet,
            }
            torch.save(checkpoint, os.path.join(config.model_path, 'epoch%d.pkl' % (epoch + 1)))


        # 测试计时
        tic = datetime.datetime.now()

        # 每个epoch测试一次acc
        # 进入测试模式
        net.eval()
        image_paths = tnscui_dataset.get_img_list()
        print(len(image_paths))
        IOU_list = []
        DICE_list = []
        for train_img in image_paths:
            train_gt = train_img.replace('image', 'mask')
            # 读取训练图片和标签图片
            image = cv2.imread(train_img)
            label = cv2.imread(train_gt)
            # 将数据转为单通道的图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            # 统一图片大小
            image = cv2.resize(image, (config.image_size, config.image_size))
            label = cv2.resize(label, (config.image_size, config.image_size))
            # 还原成4维(batch还有1维)
            image = image.reshape(1, 1, image.shape[0], image.shape[1])
            label = label.reshape(1, 1, label.shape[0], label.shape[1])
            # 处理标签，将像素值为255的改为1
            if label.max() > 1:
                label = label / 255
            # 转为tensor
            img_tensor = torch.from_numpy(image)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device, dtype=torch.float32)
            pred = net(img_tensor)
            pred = np.array(pred.data.cpu()[0])[0]
            bin_pred = (pred > 0.5).astype(np.float)
            iou = getIOU(bin_pred, label)
            dice = getDSC(bin_pred, label)
            IOU_list.append(iou)
            DICE_list.append(dice)

        IOU_final = np.mean(IOU_list)
        DICE_final = np.mean(DICE_list)
        myprint(config.record_file, 'IOU_final: %s' % IOU_final)
        myprint(config.record_file, 'DICE_final: %s' % DICE_final)
        writer.add_scalars('Acc', {'iou': IOU_final}, (epoch + 1))
        writer.add_scalars('Acc', {'dice': DICE_final}, (epoch + 1))
        # 计时结束
        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "per epoch test cost Time %02d h:%02d m:%02d s" % (h, m, s)
        myprint(config.record_file, time_str)


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--Task_name', type=str, default='test_unet')
    parser.add_argument('--data_path', type=str, default='G:/graduation project/TN-SCUI2020/my_code/1_or_data/image')
    parser.add_argument('--image_size', type=int, default=96)  # 网络输入img的size, 即输入会被强制resize到这个大小
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--save_model_step', type=int, default=2)   # 保存模型间隔
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--last_epoch', type=int, default=8)    # 为0则不续训练
    config = parser.parse_args()

    # 检查文件目录
    config.result_path = os.path.join(config.result_path, config.Task_name)
    print(config.result_path)
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)

    # 记录训练配置
    f = open(os.path.join(config.result_path, 'config.txt'), 'w')
    for key in config.__dict__:
        print('%s: %s' % (key, config.__getattribute__(key)), file=f)
    f.close()

    # 记录训练过程
    config.record_file = os.path.join(config.result_path, 'record.txt')
    f = open(config.record_file, 'a')
    f.close()

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1。
    train_net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    train_net.to(device)
    train(train_net, device, config)
