# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration



class BioT5(nn.Module):
    def __init__(self, config = None):
        super(BioT5, self).__init__()
        if(config == None):
            config = {
                "name":"biot5",
                "ckpt":"SLM4DDI/ckpts/text_ckpts/biot5"
            }
        self.config = config
        self.config.ckpt = "SLM4DDI/ckpts/text_ckpts/biot5"

        self.main_model = T5ForConditionalGeneration.from_pretrained(self.config.ckpt)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.ckpt)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size
        
    def forward(self, input_ids, attention_mask, labels):
        return self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ).loss
    
    def forward_encoding(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            #encoder_attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
    
    def encode(self, mol):
        h = self.main_model.encoder(**mol).last_hidden_state
        return h
    
    def decode(self, input_ids, attention_mask, num_beams, max_length):
        
        outputs = self.main_model.generate(
            input_ids = input_ids,  
            attention_mask = attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def decode_encoding(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            #attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)
