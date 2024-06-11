# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import re
import numpy as np
import random
import json


class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return self.data_shape


class MtiDataset(BaseDataset):
    def __init__(self, dataframe = None, split=None, tokenizer_org = None, args = None):
        if(dataframe is None):
            data_path = args.dataset
            if data_path.endswith('.pkl'):
                self.data = pd.read_pickle(data_path)
            elif data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.endswith('.txt'):
                self.data = pd.read_table(data_path)
            elif data_path.endswith('.json'):
                self.data = pd.read_json(data_path)
            else:
                raise ValueError(f'Unsupported file extension in: {data_path}')
        else:
            self.data = dataframe
            
        self.split = split
        self.tokenizer_org = tokenizer_org
        self.args = args
        self.script_mode = args.script_mode
        self.task = args.task
        
        super(MtiDataset, self).__init__(config=None)    
    
    def _load_data(self):
        self.tasks = []
        self.input = []
        self.output = []
        self.task = []
        self.ids = []
        self.head_types = []
        self.tail_types = []
        self.head_desc = []
        self.tail_desc = []
        self.heads = []
        self.tails = []
        
        if(self.split == 'all'):
            self.data = self.data
        elif(self.split == 'train'):
            self.data = self.data[self.data['split']=='train']
        elif(self.split == 'valid'):
            self.data = self.data[self.data['split']=='valid']
        elif(self.split == 'test'):
            self.data = self.data[self.data['split']=='test']
            
        if(self.args.shot == 'all'):
            filtered_data = self.data
        elif(self.args.shot == 'com'):
            filtered_data = self.data[self.data['Frequency']=='Common']
        elif(self.args.shot == 'few'):
            filtered_data = self.data[self.data['Frequency']=='Fewer']
        elif(self.args.shot == 'rare'):
            filtered_data = self.data[self.data['Frequency']=='Rare']
        
        self.data_shape = len(filtered_data)
        # col_1 = f"drug1_{self.args.sub}"
        # col_2 = f"drug2_{self.args.sub}"
        
        col_1 = f"drug1_cluster_{self.args.sub}"
        col_2 = f"drug2_cluster_{self.args.sub}"
        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.output.append(row["output"])
            self.head_types.append(row[col_1])
            self.tail_types.append(row[col_2])
            self.heads.append(row['drug1_selfies'])
            self.tails.append(row['drug2_selfies'])
            self.head_desc.append(row['drug1_description'])
            self.tail_desc.append(row['drug2_description'])
            self.ids.append(row['id'])
                    
    def __getitem__(self, i):
        mol_label_data = {}
        if(self.args.input_modal == 'type_selfies'):
            input_str = f"Please determine the type of reaction between two drugs, give me the number between 1 and 113. The drug A is a {self.head_types[i]} drug, the SELFIES is {self.heads[i]}. The drug B is  a {self.tail_types[i]} drug, the SELFIES is {self.tails[i]}"
        elif(self.args.input_modal == 'desc'):
            input_str = f"Please determine the type of reaction between two drugs, give me the number between 1 and 113. The drug {self.head_desc[i]}. The drug B {self.tail_desc[i]}"
        elif(self.args.input_modal == 'type_desc'):
            input_str = f"Please determine the type of reaction between two drugs, give me the number between 1 and 113. The drug A is a {self.head_types[i]} drug, the {self.head_desc[i]}. The drug B is a {self.tail_types[i]} drug, the {self.tail_desc[i]}"
        elif(self.args.input_modal == 'selfies'):
            input_str = f"Please determine the type of reaction between two drugs, give me the number between 1 and 113. The drug A SELFIES is {self.heads[i]}. The drug B SELFIES is {self.tails[i]}"

        mol_label_data['input'] = self.tokenizer_org(
                input_str,
                return_tensors="pt"
            )
        
        mol_label_data['output'] = self.tokenizer_org(
                str(self.output[i]),
                return_tensors="pt"
            )
        mol_label_data['id'] = self.ids[i]
        mol_label_data['truth'] = self.output[i]

        return mol_label_data



            



    
