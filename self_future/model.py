import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel

class ERC_model(nn.Module):
    def __init__(self, model_type, clsNum, last):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Model Setting"""
        condition_token = ['<s1>', '<s2>', '<s3>'] # 최대 3명
        special_tokens = {'additional_special_tokens': condition_token}
        
        # model_path = '/data/project/rw/rung/model/'+model_type
        model_path = model_type
        if model_type == 'roberta-large':
            self.model = RobertaModel.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif model_type == 'bert-large-uncased':
            self.model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            print('error')
        tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(tokenizer))
        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input_tokens, batch_cls_positions):
        """
            batch_input_tokens: (batch, len)
        """
        batch_context_output = []
        batch_context_output_all = self.model(batch_input_tokens).last_hidden_state # (batch, token_num, 1024)
        for batch in range(batch_context_output_all.shape[0]):
            cls_position = batch_cls_positions[batch]
            context_output = batch_context_output_all[batch, cls_position, :] # (1024)
            batch_context_output.append(context_output.unsqueeze(0)) # [(1, 1024)]
        batch_context_outputs = torch.cat(batch_context_output, 0) # (batch, 1024)
        
        context_logit = self.W(batch_context_outputs) # (batch, clsNum)
        
        return context_logit