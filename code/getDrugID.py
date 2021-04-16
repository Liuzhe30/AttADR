#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 8:41
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : getDrugID.py
# @Software: PyCharm

import os
import sys
import pandas as pd
import json

print(sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0]))
from dbutils import DBconnection

db_DrugKB = DBconnection("DrugKB")   # connect db:DrugKB
print("MongoDB connected successfully!")

project_path = os.path.abspath(__file__).replace('\\', '').rsplit('/', 2)[0]

# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
def get_drug_id(file):
    # 获取药物名的ID
    drugList = []
    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data:
            if item['id_A'] not in drugList:
                drugList.append(item['id_A'])

            if item['id_B'] not in drugList:
                drugList.append(item['id_B'])
        # 升序排列
        drugList.sort()
        print(len(drugList))
        # 写入txt
        # save_path = project_path + '/data/DrugID.txt'
        # text_save(save_path,drugList)
        return drugList

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

# ------------------------------------------------------------------------------------
def get_target_drug_id(drug_list):
    drugList = []
    for drugItem in db_DrugKB.mdbi['drug'].find().batch_size(5):
        # drugDict = {
        #     'id_A': drugItem['ddi'][0],
        #     'drugA': drugItem['ddi-name'][0],
        #     'id_B': drugItem['ddi'][1],
        #     'drugB': drugItem['ddi-name'][1]
        #     }

        # if drugDict['drugA'] in drug_list:
        #     drugList.append([drugDict['id_A'], drugDict['drugA']])
        #     print(drugDict['id_A'], drugDict['drugA'])
        #
        # if drugDict['drugB'] in drug_list:
        #     drugList.append([drugDict['id_B'], drugDict['drugB']])
        #     print(drugDict['id_B'], drugDict['drugB'])
        #
        # li = []
        # for item in drugList:
        #     if item not in li:
        #         li.append(item)

        drugDict = {
           'id': drugItem['_id'],
           'drug': drugItem['name']}

        if drugDict['id'] in drug_list:
            drugList.append([drugDict['id'], drugDict['drug']])
            print(drugDict['id'], drugDict['drug'])

    print(len(drugList))

    save_path = project_path + '/data/TargetDrugID.txt'
    text_save(save_path,drug_list)

# =============================================================================================
if __name__ == '__main__':
    # get Drug ID
    ddi_path = project_path + '/data/ddi.json'
    get_drug_id(ddi_path)

    # # Get Target Drug ID
    # target_path = project_path + '/data/DrugTarget.csv'
    # drug_list = read_csv_file(target_path)
    # get_target_drug_id(drug_list)


