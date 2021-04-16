#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 14:57
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : nlp.py
# @Software: PyCharm



# =================================================================================================================
import pandas as pd
import stanfordnlp
# stanfordnlp.download('en')
import numpy as np
import os
import json
import csv
import codecs


def NLPProcess(druglist,df_interaction):
    def addMechanism(node):
        if int(sonsNum[int(node-1)])==0:
            return
        else:
            for k in sons[node-1]:
                if int(k)==0:
                    break
                if dependency[int(k-1)].text == drugA[i] or dependency[int(k-1)].text == drugB[i]:
                    # continue
                    break
                quene.append(int(k))
                addMechanism(int(k))
        return quene

    nlp = stanfordnlp.Pipeline()
    event=df_interaction
    print(event)
    mechanism=[]
    action=[]
    drugA=[]
    drugB=[]
    for i in range(len(event)):
        doc=nlp(event[i])
        print(doc.sentences[0])
        dependency = []
        for j in range(len(doc.sentences[0].words)):
            dependency.append(doc.sentences[0].words[j])
        sons=np.zeros((len(dependency),len(dependency)))
        sonsNum=np.zeros(len(dependency))
        flag=False
        count=0
        for j in dependency:
            if j.dependency_relation=='root':
                root=int(j.index)
                action.append(j.lemma)
            if j.text in druglist:
                if count<2:
                    if flag==True:
                        drugB.append(j.text)
                        count+=1
                    else:
                        drugA.append(j.text)
                        flag=True
                        count+=1
            sonsNum[j.governor-1]+=1
            sons[j.governor-1,int(sonsNum[j.governor-1]-1)]=int(j.index)
        quene=[]
        for j in range(int(sonsNum[root-1])):
            if dependency[int(sons[root-1,j]-1)].dependency_relation=='obj' or dependency[int(sons[root-1,j]-1)].dependency_relation=='nsubj:pass':
                quene.append(int(sons[root-1,j]))
                break
        quene=addMechanism(quene[0])
        quene.sort()
        mechanism.append(" ".join(dependency[j-1].text for j in quene))
        if mechanism[i]=="the fluid retaining activities":
            mechanism[i]="the fluid"
        if mechanism[i]=="atrioventricular blocking ( AV block )":
            mechanism[i]='the atrioventricular blocking ( AV block ) activities increase'
    return mechanism,action,drugA,drugB



# =============================================================================================
if __name__ == '__main__':

    output = []

    file_path = os.path.abspath(__file__).replace('\\', '').rsplit('/', 2)[0]
    with open(file_path + '/data/new_ddi.json', 'r', encoding='utf-8') as fo:
        json_data = json.load(fo)
        for item in json_data[119687:]:
        # for i,item in enumerate(json_data):
        #     if item['description'] == 'The risk or severity of adverse effects can be increased when Pomalidomide is combined with Aripiprazole lauroxil.':
        #         print(i)
            # 获取药物名的ID
            drugList = []
            drugList.append(item['id_A'])
            drugList.append(item['id_B'])

            # 获取药物的Description
            description = item['description'].lower()  # 大写字母转小写
            print(description)

            description = description.replace(item['drugA'],item['id_A']).replace(item['drugB'],item['id_B']).replace('.','')
            print(description)

            new_description = description.replace(item['id_A'], 'DrugA').replace(item['id_B'], 'DrugB')
            print(new_description)

            result = NLPProcess(['DrugA', 'DrugB'], [new_description])

            if result[2][0] == 'DrugA' and result[3][0] == 'DrugB':
                result[2][0] = item['id_A']
                result[3][0] = item['id_B']

            else:
                result[2][0] = item['id_B']
                result[3][0] = item['id_A']

            new_output = [result[0][0], result[1][0], result[2][0], result[3][0]]
            # output.append(new_output)
            print(new_output)

            with open(file_path + '/data/ddi/' + result[2][0] + '_' + result[3][0] + '.json', "w") as w:
                json.dump(new_output, w, indent=4)

    # # 获取药物的Description
    # item = {
    #     'drugA': 'amphetamine',
    #     'id_A': 'DB00182',
    #     'drugB': 'methylenedioxyethamphetamine',
    #     'id_B': 'DB01566'
    # }
    # # description = 'the metabolism of phenobarbital can be increased when combined with barbital.'.lower()  # 大写字母转小写
    # # print(description)
    #
    # # description = description.replace(item['drugA'], item['id_A']).replace(item['drugB'], item['id_B']).replace('.', '')
    # # print(description)
    #
    # description = 'DB00182 may increase the hypertensive activities of DB01566'
    #
    # new_description = description.replace(item['id_A'], 'DrugA').replace(item['id_B'], 'DrugB')
    # print(new_description)
    #
    # result = NLPProcess(['DrugA', 'DrugB'], [new_description])
    #
    # if result[2][0] == 'DrugA' and result[3][0] == 'DrugB':
    #     result[2][0] = item['id_A']
    #     result[3][0] = item['id_B']
    #
    # else:
    #     result[2][0] = item['id_B']
    #     result[3][0] = item['id_A']
    #
    # new_output = [result[0][0], result[1][0], result[2][0], result[3][0]]
    # # output.append(new_output)
    # print(new_output)
    #
    # with open(file_path + '/data/ddi/' + result[2][0] + '_' + result[3][0] + '.json', "w") as w:
    #     json.dump(new_output, w, indent=4)
    #
    #
    #
    #

