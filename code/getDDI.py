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
import json
import founction

print(sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0]))
from dbutils import DBconnection

db_DrugKB = DBconnection("DrugKB")   # connect db:DrugKB
print("MongoDB connected successfully!")

# ===================================================================================
def query_drug(index_path, savePath):

    drugList = founction.read_file(index_path)
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

# =============================================================================================
if __name__ == '__main__':

    index_path = "/home/cqfnenu/attddi-Qiufen/data/DrugTarget.csv"
    save_path = "/home/cqfnenu/attddi-Qiufen/data/"

    query_drug(index_path, save_path)









