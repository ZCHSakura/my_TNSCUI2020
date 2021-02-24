# coding:utf-8
import csv
import os


with open('../dataprocess/segandclassifydata_zch.csv', 'w', encoding="utf-8", newline='') as file_obj:
    writer = csv.writer(file_obj)
    row = ['label', 'Image', 'Mask']
    writer.writerow(row)
    for i in range(2500):
        for j in range(2):
            img = os.path.abspath('../..') + '\\augtrain\\image\\{0}_{1}.bmp'.format(i, j + 1)
            mask = os.path.abspath('../..') + '\\augtrain\\mask\\{0}_{1}.bmp'.format(i, j + 1)
            row = ['0', img, mask]
            writer.writerow(row)

with open('../dataprocess/segandclassifydata_zch.csv', 'r', encoding="utf-8", newline='') as file_obj:
    reader = csv.reader(file_obj)
    for i in reader:
        print(i)

