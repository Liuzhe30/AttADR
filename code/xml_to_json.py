#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 18:35
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : xml_to_json.py
# @Software: PyCharm

# In[1]:
# !/usr/bin/python3

import untangle
import pandas as pd
import numpy as np
import os


filename = "/home/cqfnenu/attddi-Qiufen/data/full_database.xml"
obj = untangle.parse(filename)

# Building dataframe of chemical descriptors
# Data Frame of DrugBank Small Molecule Type Drugs
df_drugbank_sm = pd.DataFrame(
    # columns=["drugbank_id", "name", "cas", "smiles", "logP ALOGPS", "logP ChemAxon", "solubility ALOGPS",
    #          "pKa (strongest acidic)", "pKa (strongest basic)"])
    columns=["drugbank_id", "name", "cas", "smiles"])

i = -1
# iterate over drug entries to extract information
for drug in obj.drugbank.drug:
    drug_type = str(drug["type"])

    # select for small molecule drugs
    if drug_type in ["small molecule", "Small Molecule", "Small molecule"]:
        i = i + 1

        # Get drugbank_id
        for id in drug.drugbank_id:
            if str(id["primary"]) == "true":
                df_drugbank_sm.loc[i, "drugbank_id"] = id.cdata
        # Drug name
        df_drugbank_sm.loc[i, "name"] = drug.name.cdata

        # Drug CAS
        df_drugbank_sm.loc[i, "cas"] = drug.cas_number.cdata

        # Get SMILES, logP, Solubility
        # Skip drugs with no structure. ("DB00386","DB00407","DB00702","DB00785","DB00840",
        #                                            "DB00893","DB00930","DB00965", "DB01109","DB01266",
        #                                           "DB01323", "DB01341"...)
        if len(drug.calculated_properties.cdata) == 0:  # If there is no calculated properties
            continue
        else:
            for property in drug.calculated_properties.property:
                if property.kind.cdata == "SMILES":
                    df_drugbank_sm.loc[i, "smiles"] = property.value.cdata

                # if property.kind.cdata == "logP":
                #     if property.source.cdata == "ALOGPS":
                #         df_drugbank_sm.loc[i, "logP ALOGPS"] = property.value.cdata
                #     if property.source.cdata == "ChemAxon":
                #         df_drugbank_sm.loc[i, "logP ChemAxon"] = property.value.cdata
                #
                # if property.kind.cdata == "Water Solubility":
                #     df_drugbank_sm.loc[i, "solubility ALOGPS"] = property.value.cdata
                #
                # if property.kind.cdata == "pKa (strongest acidic)":
                #     df_drugbank_sm.loc[i, "pKa (strongest acidic)"] = property.value.cdata
                #
                # if property.kind.cdata == "pKa (strongest basic)":
                #     df_drugbank_sm.loc[i, "pKa (strongest basic)"] = property.value.cdata


df_drugbank_sm.head(10)

print(df_drugbank_sm.shape)

# # Drop drugs without SMILES from the dataframe
# df_drugbank_smiles = df_drugbank_sm.dropna()
# df_drugbank_smiles = df_drugbank_smiles.reset_index(drop=True)
# print(df_drugbank_smiles.shape)

# df_drugbank_smiles.head()

# write to csv
df_drugbank_sm.to_csv("/home/cqfnenu/attddi-Qiufen/data/smiles_1.csv", encoding='utf-8', index=False)

