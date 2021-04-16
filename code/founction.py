#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 0:07
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : founction.py
# @Software: PyCharm

import pandas as pd
import json
import os

# ------------------------------------------------------------------------------------
def json_to_csv(dir_path, save_path):
    file_num = 0
    ddi_list = []

    for (root, dirs, files) in os.walk(dir_path):
        for file_name in files:
           file_num = file_num + 1

           with open(os.path.join(root, file_name), 'r') as fo:
               data = json.load(fo)
               ddi_list.append(data)

    print(len(ddi_list))
    print("ddi文件夹里的文件个数为：", file_num)

    name = ['mechanis', 'action', 'drugA','drugB']
    test = pd.DataFrame(columns=name, data=ddi_list)
    print(test)
    test.to_csv(save_path, encoding='gbk')

# ------------------------------------------------------------------------------------
def text_save(filename, data):
    # list data to txt file
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'", '').replace(',','').replace('(','').replace(')','') + '\n'
        file.write(s)
    file.close()
    print("Done!")

# ------------------------------------------------------------------------------------
def read_csv_file(index_path):
    """ purpose: csv to List """
    drug = []
    df = pd.read_csv(index_path, header=0, usecols=[2])
    df = df.values.tolist()
    for item in df:
        drug.append(item[0])
    print(len(drug))
    return drug

# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    file_path = '/home/cqfnenu/attddi-Qiufen/data/ddi'
    out_path = '/home/cqfnenu/attddi-Qiufen/data/event.csv'

    json_to_csv(file_path, out_path)

    print("*************************************************")