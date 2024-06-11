# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models import SUPPORTED_Model, SUPPORTED_CKPT
from transformers import T5Tokenizer
import torch.nn.functional as F
import numpy as np


class MtiModel(nn.Module):
    def __init__(self, config=None):
        super(MtiModel, self).__init__()
        self.config = config
        self.config.hidden_size = 768
        self.language_model  = SUPPORTED_Model[self.config.model_name](self.config)
        self.tokenizer = self.language_model.tokenizer

        if(self.config.freeze == True or self.config.script_mode == 'data_label_classification' or self.config.script_mode == 'data_label'):
            for k, v in self.language_model.named_parameters():
                v.requires_grad = False
                
        
    def forward(self, mol):
        h = mol['input']['input_ids']
        h_attention_mask = mol['input']['attention_mask']
        output = self.language_model(
            input_ids = h,
            attention_mask = h_attention_mask,
            labels = mol["output"]["input_ids"]
        )
        
        return output
    
    def embeddings(self, mol):
        embeddings = self.text_encode(mol)
        h = embeddings.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2)
        return h
    

    def encoding(self, mol):
        input_encoding = self.text_encode(mol)
        return input_encoding
    
    def text_encode(self, mol):
        h = self.language_model.encode(mol['input'])
        return h
    
    def generate_text(self, mol):
        h = mol['input']['input_ids']
        h_attention_mask = mol['input']['attention_mask']
        text = self.language_model.decode(
                input_ids = h, 
                attention_mask = h_attention_mask, 
                num_beams = 1,
                max_length = 5
            )
        return text
