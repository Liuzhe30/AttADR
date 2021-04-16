#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 18:35
# @Author  : Qiufen.Chen
# @Email   : 1760812842@qq.com
# @File    : parseDrugbank.py
# @Software: PyCharm

"""purpose: parse drugbank and get smiles datas"""

import untangle
import pandas as pd
import os

file_path = os.path.abspath(__file__).replace('\\', '').rsplit('/', 2)[0]

def getSmile(xml_file):
    obj = untangle.parse(xml_file)

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
    df_drugbank_sm.to_csv(file_path + "/data/smiles.csv", encoding='utf-8', index=False)

# =======================================================================================
if __name__ == "__main__":
    filename = file_path + "/data/full_database.xml"
    getSmile(filename)