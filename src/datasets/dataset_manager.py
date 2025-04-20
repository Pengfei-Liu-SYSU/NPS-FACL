# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import os
import csv
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import torch.nn as nn
import re
import numpy as np
import random

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT
from rdkit import Chem
import re

class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
    #     self._load_data()
        
    # @abstractmethod
    # def _load_data(self):
    #     raise NotImplementedError

    def __len__(self):
        return self.data_shape

class MainDataset(BaseDataset):
    def __init__(self, split='train', fold=0, tokenizer_org=None, args=None):
        data_path = args.dataset
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        elif data_path.endswith('.jsonl'):
            self.data = pd.read_json(data_path, lines=True)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.tokenizer_org = tokenizer_org
        self.split = split  
        self.fold = fold  # Fold number (0 to 4 for 5-fold CV)
        self.args = args
   
        # Split data into train+val (80%) and test (20%) as a holdout set
        train_val_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=args.random_state, stratify=self.data["NPS"]
        )
        
        if split == "test":
            self.data = test_data
        else:
            # Perform 5-fold cross-validation on train+val data
            kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
            train_val_splits = list(kf.split(train_val_data))
            
            # Select the current fold
            train_idx, val_idx = train_val_splits[fold]
            train_data = train_val_data.iloc[train_idx]
            val_data = train_val_data.iloc[val_idx]
            
            if split == "train":
                self.data = train_data
            elif split == "val":
                self.data = val_data
        
        self.data_shape = len(self.data)

        self.input = []
        self.output = []

        for _, row in tqdm(self.data.iterrows(), total=self.data_shape, desc=f"Loading {split} dataset (Fold {fold})"):
            if self.args.model_name == 'biot5':
                self.input.append(row["selfies"])
            else:
                self.input.append(row["smiles"])  
            self.output.append(row["NPS"])

        super(MainDataset, self).__init__(config=None)
                                 
    def __len__(self):
        return self.data_shape

    def __getitem__(self, i):

        data = {}

        smiles_text = str(self.input[i])
        label = torch.tensor(self.output[i], dtype=torch.long)  

        # **Tokenization**
        tokenized_smiles = self.tokenizer_org(
            smiles_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )

        data["input_ids"] = tokenized_smiles["input_ids"].squeeze(0)
        data["attention_mask"] = tokenized_smiles["attention_mask"].squeeze(0)
        data["output"] = label
        data["smiles"] = smiles_text
        return data
