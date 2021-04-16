#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 15:41
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : getTargetSmiles.py
# @Software: PyCharm


import csv
import json
import codecs
import pandas as pd

# ---------------------------------------------------------------------------
def csv_to_json(csvfile):
    # Python implement csv format conversion to json format file
    smileList = []
    with open(csvfile, 'r', encoding='utf-8-sig') as file:
        smile = csv.DictReader(file)
        for i in smile:
            dict = {}
            for k, v in i.items():
                dict[k] = v

            # print(dict)
            smileList.append(dict)

    return smileList
        # save_path = '/home/cqfnenu/attddi-Qiufen/data/'
        # with open(save_path + 'targetSmiles.json', "w") as w:
        #     json.dump(smileList, w, indent=4)

# ---------------------------------------------------------------------------
def get_target_smiles(smile_list):

    txtfile = '/home/cqfnenu/attddi-Qiufen/data/DrugID.txt'

    with open(txtfile, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
        drugID = [line.replace('\n','') for line in lines]

        # print(len(drugID))
        # print(drugID)

        smile = []
        idList = []
        for element in smile_list:
            if element['drugbank_id'] in drugID:
                idList.append(element['drugbank_id'])
                smile.append(element)

        print(idList)

        # Achieve deweighting of the same element in two list
        new_drugID = [i for i in drugID if i not in idList]
        for every in new_drugID:
            smile.append({
                'drugbank_id': every,
                'smiles': 'Null'
            })
        print(len(smile))

        save_path = '/home/cqfnenu/attddi-Qiufen/data/smiles.csv'
        # with open(save_path + 'smiles.json', "w") as w:
        #     json.dump(smile, w, indent=4)

        # 解决写入csv时中文乱码，用'utf_8_sig'
        f = codecs.open(save_path, 'w', 'utf_8_sig')
        writer = csv.writer(f)
        for item in smile:
            writer.writerow([item['drugbank_id'], item['smiles']])
        f.close()


# =====================================================================================
if __name__ == '__main__':

    csv_path = '/home/cqfnenu/attddi-Qiufen/data/AllSmiles.csv'
    smile_list = csv_to_json(csv_path)
    get_target_smiles(smile_list)
