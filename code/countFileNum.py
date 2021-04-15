#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 14:46
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : countFileNum.py
# @Software: PyCharm

import os
import json
import csv
import codecs
import pandas as pd

def text_save(filename, data):
    # 将列表数据写入txt文件
    file = open(filename,'a')
    for i in range(len(data)):
        # 去除[],这两行按数据不同，可以选择
        s = str(data[i]).replace('[','').replace(']','')
        # 去除单引号，逗号，每行末尾追加换行符
        s = s.replace("'", '').replace(',','').replace('(','').replace(')','') + '\n'
        file.write(s)
    file.close()
    print("保存文件成功")
# ----------------------------------------------------------------------
def json_to_csv(dir_path, save_path):
    file_num = 0
    ddi_list = []

    for (root, dirs, files) in os.walk(dir_path):  #列出windows目录下的所有文件和文件名
        for file_name in files:
           file_num = file_num + 1

           with open(os.path.join(root, file_name), 'r') as fo:
               data = json.load(fo)
               ddi_list.append(data)

    print(len(ddi_list))
    print("ddi文件夹里的文件个数为：", file_num)

    name = ['mechanis', 'action', 'drugA','drugB']
    test = pd.DataFrame(columns=name, data=ddi_list)  # 数据有三列，列名分别为one,two,three
    print(test)
    test.to_csv(save_path, encoding='gbk')

# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    file_path = '/home/cqfnenu/attddi-Qiufen/data/ddi'
    out_path = '/home/cqfnenu/attddi-Qiufen/data/event.csv'

    json_to_csv(file_path, out_path)

    print("*************************************************")