#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 10:14
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : queryDrugKB.py
# @Software: PyCharm

import os
import sys
import pandas as pd
import numpy as np
import json

print(sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0]))
from dbutils import DBconnection

# connect db:DrugKB
db_DrugKB = DBconnection("DrugKB")
print("MongoDB connected successfully!")

# ==================================================================================================
txtfile = '/home/cqfnenu/attddi-Qiufen/data/DrugID.txt'

def read_file(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        drugID = [line.replace('\n', '') for line in lines]
    return drugID

# ==================================================================================================
def query_target(drug_id, save_path):
    drugList = []
    for element in drug_id:
        result = db_DrugKB.mdbi['drug'].find({'_id': element})
        for item in result:
            drug_dict = {'id': item['_id'],
                         'drug': item['name']}
            if 'targets' in item.keys():
                target_list = []
                for array_item in item['targets']:
                        target_dict = {'target-name': array_item['name'],
                                       'target-id': array_item['id']}
                        target_list.append(target_dict)
                drug_dict.update({'target': target_list})
                # print(drug_dict)

            else:
                drug_dict.update({'target': 'Null'})
            drugList.append(drug_dict)
    print(len(drugList))

    with open(save_path + 'targets.json', "w") as w:
        json.dump(drugList, w, indent=4)

# ==================================================================================================
def query_enzyme(drug_id, save_path):

    drugList = []
    for element in drug_id:
        result = db_DrugKB.mdbi['drug'].find({'_id': element})
        for item in result:
            drug_dict = {'id': item['_id'],
                         'drug': item['name']}
            if 'enzymes' in item.keys():
                enzyme_list = []
                for array_item in item['enzymes']:
                        enzyme_dict = {'enzyme-name': array_item['name'],
                                       'enzyme-id': array_item['id']}
                        enzyme_list.append(enzyme_dict)
                drug_dict.update({'enzyme': enzyme_list})
                # print(drug_dict)

            else:
                drug_dict.update({'enzyme': 'Null'})
            drugList.append(drug_dict)
    print(len(drugList))

    with open(save_path + 'enzymes.json', "w") as w:
        json.dump(drugList, w, indent=4)

# ==================================================================================================
def query_pathway(drug_id, save_path):
    drugList = []
    for element in drug_id:
        result = db_DrugKB.mdbi['drug'].find({'_id': element})
        for item in result:
            drug_dict = {'id': item['_id'],
                         'drug': item['name']}

            if 'pathways' in item.keys():
                drug_dict.update({'pathway-id': item['pathways']['smpdb-id'],
                                  'pathway-name': item['pathways']['name']})
                print(drug_dict)

            else:
                drug_dict.update({'pathway-id': 'Null',
                                  'pathway-name': 'Null'})
            drugList.append(drug_dict)
    print(len(drugList))

    with open(save_path + 'pathways.json', "w") as w:
        json.dump(drugList, w, indent=4)

# ==================================================================================================
def query_transporter(drug_id, save_path):
    transporter = {}
    for compound in db_DrugKB.mdbi['transporter'].find():
        transporter.update({compound['compound-drug-id']: {'transporter': compound['Transporter']}})

    drugList = []
    for element in drug_id:
        result = db_DrugKB.mdbi['drug'].find({'_id': element})
        for item in result:

            if item['_id'] in transporter.keys():
                drugList.append({'id': item['_id'],
                                 'drug': item['name'],
                                 'transporter': transporter[item['_id']]['transporter']})

            else:
                drugList.append({'id': item['_id'],
                                 'drug': item['name'],
                                 'transporter': 'Null'})
    print(len(drugList))

    with open(save_path + 'transporters.json', "w") as w:
        json.dump(drugList, w, indent=4)


# ===================================================================================
index_path = '/home/cqfnenu/attddi-Qiufen/data/DrugID.txt'
save_path = '/home/cqfnenu/attddi-Qiufen/data/'

drug_name = read_file(index_path)

query_target(drug_name, save_path)
query_enzyme(drug_name, save_path)
query_pathway(drug_name,save_path)
query_transporter(drug_name, save_path)
