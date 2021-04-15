#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/15 19:24
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : index_drug.py
# @Software: PyCharm
# @Update  : 2021/3/25 16:00

""" Queries with ddi data indexed with MongoDB """

import os
import sys
import pandas as pd
import json

print(sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0]))
from dbutils import DBconnection

db_DrugKB = DBconnection("DrugKB")   # connect db:DrugKB
print("MongoDB connected successfully!")

# ===================================================================================
def read_file(index_path):
    """ purpose:  csv to List """

    drug = []
    df = pd.read_csv(index_path, header=0, usecols=[2])
    df = df.values.tolist()
    for item in df:
        drug.append(item[0])
    print(len(drug))
    return drug

# ===================================================================================
def query_drug(drugList, savePath):

    ddiList = []

    for ddiItem in db_DrugKB.mdbi['ddi'].find().batch_size(5):
        drugDict = {
                     'id_A': ddiItem['ddi'][0],
                     'drugA': ddiItem['ddi-name'][0],
                     'id_B': ddiItem['ddi'][1],
                     'drugB': ddiItem['ddi-name'][1],
                     'description': ddiItem['ddi-drugbank']['description'],
                     'database': 'drugbank'}

        if (drugDict['id_A'] in drugList) or \
            (drugDict['id_B'] in drugList):
            ddiList.append(drugDict)
            # print(drugDict)
    print(len(ddiList))

    with open(savePath + 'ddi.json', "w") as w:
        json.dump(ddiList, w, indent=4)


# ===================================================================================
index_path = "/home/cqfnenu/attddi-Qiufen/data/DrugTarget.csv"
save_path = "/home/cqfnenu/attddi-Qiufen/data/"
drug_name = read_file(index_path)
query_drug(drug_name, save_path)









