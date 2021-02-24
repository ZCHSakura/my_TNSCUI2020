# coding:utf-8
import csv
import os
import numpy as np


with open('G:/graduation project/TN-SCUI2020/my_code/1_or_data/all_id_cate.csv', 'w', encoding="utf-8", newline='') as file_obj:
    writer = csv.writer(file_obj)
    row = ['ID', 'CATE']
    writer.writerow(row)
    for i in range(0, 3644):
        for j in range(2):
            idd = '{0}_{1}'.format(i, j + 1)
            row = [idd, np.random.randint(0, 2)]
            writer.writerow(row)

with open('G:/graduation project/TN-SCUI2020/my_code/1_or_data/all_id_cate.csv', 'r', encoding="utf-8", newline='') as file_obj:
    reader = csv.reader(file_obj)
    for i in reader:
        print(i)

