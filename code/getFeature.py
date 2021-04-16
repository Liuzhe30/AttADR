#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 14:55
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : getFeature.py
# @Software: PyCharm

import os
import json
import numpy as np
import founction
import pandas as pd

file_path = os.path.abspath(__file__).replace('\\', '').rsplit('/', 2)[0]

# ===================================================================================
def get_pathway(file):
    pathway = []
    drugList = []

    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data:
            if item['pathway-id'] != 'Null' and item['pathway-id'] not in pathway:
                pathway.append(item['pathway-id'])

            if item['id'] not in drugList:
                drugList.append(item['id'])

        print('Total pathways: ', len(pathway))
        print('Total drugs: ', len(drugList))

        pathway.sort()
        drugList.sort()

        a = np.zeros((len(drugList), len(pathway)))
        pathway_dict = dict((c, i) for i, c in enumerate(pathway))
        drug_dict = dict((c, i) for i, c in enumerate(drugList))

        for element in json_data:
            if element['pathway-id'] != 'Null':
                a[drug_dict[element['id']]][pathway_dict[element['pathway-id']]] = 1

        np.save(file_path + '/data/numpy/pathway.npy', a)
        founction.text_save(file_path+ '/data/PathwayList.txt', pathway)

        return a


# ===================================================================================
def get_targets(file):
    target = []
    drugList = []
    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data:
            if item['target'] != 'Null':
                for element in item['target']:
                    if element['target-id'] not in target:
                        target.append(element['target-id'])
            if item['id'] not in drugList:
                drugList.append(item['id'])

        print('Total targets: ', len(target))
        print('Total drugs: ', len(drugList))

        target.sort()
        drugList.sort()

        b = np.zeros((len(drugList), len(target)))
        target_dict = dict((c, i) for i, c in enumerate(target))
        drug_dict = dict((c, i) for i, c in enumerate(drugList))

        for element in json_data:
            if element['target'] != 'Null':
                for every in item['target']:
                    b[drug_dict[element['id']]][target_dict[every['target-id']]] = 1

        np.save(file_path + '/data/numpy/target.npy', b)
        founction.text_save(file_path + '/data/TargetList.txt', target)
        return b

# ===================================================================================
def get_enzymes(file):
    drugList = []
    enzyme = []
    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data:
            if item['enzyme'] != 'Null':
                for element in item['enzyme']:
                    if element['enzyme-id'] not in enzyme:
                        enzyme.append(element['enzyme-id'])

            if item['id'] not in drugList:
                drugList.append(item['id'])

        print('Total enzymes: ', len(enzyme))
        print('Total drugs: ', len(drugList))

        enzyme.sort()
        drugList.sort()
        c = np.zeros((len(drugList), len(enzyme)))
        target_dict = dict((c, i) for i, c in enumerate(enzyme))
        drug_dict = dict((c, i) for i, c in enumerate(drugList))

        for element in json_data:
            if element['enzyme'] != 'Null':
                for every in element['enzyme']:
                    # print(every)
                    c[drug_dict[element['id']]][target_dict[every['enzyme-id']]] = 1

        np.save(file_path + '/data/numpy/enzyme.npy', c)
        founction.text_save(file_path + '/data/EnzymeList.txt', enzyme)
        return c

# ===================================================================================
def get_smile(file):
    count = 0
    smile = []
    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data:
            if item['smiles'] != 'Null' and item['smiles'] not in smile:
                smile.append(item['smiles'])
                count += 1
        print('Total smiles: ', len(smile))

        smile.sort()

# ===================================================================================
def get_transporter(file):
    transporter = []
    drugList = []

    with open(file, 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        # print(len(json_data))
        for item in json_data:
            if 'transporter' in item.keys() and (item['transporter'] not in transporter) and item['transporter'] != 'Null':
               transporter.append(item['transporter'])

            if item['id'] not in drugList:
                drugList.append(item['id'])

        print('Total transporters: ', len(transporter))
        print('Total drugs: ', len(drugList))

        transporter.sort()
        drugList.sort()

        d = np.zeros((len(drugList), len(transporter)))
        transpoter_dict = dict((c, i) for i, c in enumerate(transporter))
        drug_dict = dict((c, i) for i, c in enumerate(drugList))

        for element in json_data:
            if element['transporter'] != 'Null':
                d[drug_dict[element['id']]][transpoter_dict[element['transporter']]] = 1

        np.save(file_path + '/data/numpy/transporter.npy', d)
        founction.text_save(file_path + '/data/TransporterList.txt', transporter)
        return d

# ===================================================================================
def csv_to_mol(csv_file):
    """purpose: csv file to npy file"""
    out = pd.read_csv(csv_file, header=0, usecols=[1, 3])
    # print(out.head())
    df = out.values.tolist()

    mol = []
    drug = []
    for item in df:
        # print(item)
        drug.append(item[0])

        if item[1] == "0'*167":
            item[1] = [0 for i in range(167)]
            print(item[1])
            mol.append(item[1])
        else:
            new_mol = [int(i) for i in item[1]]
            print(new_mol)
            mol.append(new_mol)
    print(len(mol))

    np.save(file_path + '/data/numpy/smile.npy', mol)


# ===================================================================================
if __name__ == '__main__':
    # -------------------------------------------------------
    pathway_file = file_path + '/data/pathways.json'
    a = get_pathway(pathway_file)

    # -------------------------------------------------------
    target_file = file_path + '/data/targets.json'
    b = get_targets(target_file)

    # -------------------------------------------------------
    enzyme_file = file_path + '/data/enzymes.json'
    c = get_enzymes(enzyme_file)

    # -------------------------------------------------------
    smile_file = file_path + '/data/smiles.json'
    get_smile(smile_file)

    # -------------------------------------------------------
    transporter_file = file_path + '/data/transporters.json'
    d = get_transporter(transporter_file)

    # -------------------------------------------------------
    smile_csv = file_path + "/data/new_smiles.csv"
    csv_to_mol(smile_csv)

    # -------------------------------------------------------
    a_1 = np.hstack((a,b))
    # a_2 = np.hstack((a_1,c))
    a_2 = np.hstack((c,d))
    a_3 = np.hstack((a_1, a_2))
    # np.save(file_path + '/data/numpy/pathway_target_enzyme.npy', a_2)
    np.save(file_path + '/data/numpy/pathway_target_enzyme_transporter.npy', a_3)
    print('Done!')


