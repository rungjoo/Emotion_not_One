import torch
from torch.nn.functional import softmax

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

roberta_tokenizer = RobertaTokenizer.from_pretrained('/data/project/rw/rung/model/roberta-large/')
bert_tokenizer = BertTokenizer.from_pretrained('/data/project/rw/rung/model/bert-large-uncased/')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('/data/project/rw/rung/model/gpt2-large/')
gpt_tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})

condition_token = ['<s1>', '<s2>', '<s3>'] # 최대 3명
special_tokens = {'additional_special_tokens': condition_token}
roberta_tokenizer.add_special_tokens(special_tokens)
bert_tokenizer.add_special_tokens(special_tokens)
gpt_tokenizer.add_special_tokens(special_tokens)

""" word embedidng """
import io
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

""" batch """
def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(ids+add_ids)
    
    return torch.tensor(pad_ids)

def encode_right_truncated_gpt(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.cls_token_id]

def padding_gpt(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids)
    
    return torch.tensor(pad_ids)

def label2gray(emotion, sentiment, gray_type, label_list, sentidict, cos_dict):
    gray_label = []
    if len(label_list) > 3:        
        if gray_type == 'heuristic':
            sim_emotion_list = sentidict[sentiment]
            neu_emotion_list = sentidict['neutral']
            for cand_emo in label_list:
                if cand_emo == emotion:
                    gray_label.append(1)
                elif cand_emo in sim_emotion_list:
                    gray_label.append(0.5)
#                 elif cand_emo in neu_emotion_list:
#                     gray_label.append(0.3)
                else:
                    gray_label.append(0)
        elif gray_type == 'word':
            for cand_emo in label_list:
                sim = cos_dict[emotion+cand_emo]
                gray_label.append(sim)
        else: # teacher
            print('Error')
            return 
    else:
        if gray_type == 'heuristic':
            for cand_senti in label_list:
                if cand_senti == sentiment:
                    gray_label.append(1)
                elif cand_senti == 'negative' and sentiment == 'positive':
                    gray_label.append(0)
                elif cand_senti == 'negative' and sentiment == 'neutral':
                    gray_label.append(0.5)
                elif cand_senti == 'positive' and sentiment == 'negative':
                    gray_label.append(0)
                elif cand_senti == 'positive' and sentiment == 'neutral':
                    gray_label.append(0.5)
                elif cand_senti == 'neutral' and sentiment == 'positive':
                    gray_label.append(0.5)
                elif cand_senti == 'neutral' and sentiment == 'negative':
                    gray_label.append(0.5)                    
        elif gray_type == 'word':
            for cand_senti in label_list:
                sim = cos_dict[sentiment+cand_senti]
                gray_label.append(sim)
        else: # teacher
            print('Error')
            return 
    
    gray_label_dist = []
    total_sum = 0
    for gray_label_value in gray_label:
        if gray_label_value > 0:
            total_sum += gray_label_value
    for gray_label_value in gray_label:
        if gray_label_value > 0:
            gray_label_dist.append(gray_label_value/total_sum)
        else:
            gray_label_dist.append(0)
    return gray_label_dist

def make_batch_roberta(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, roberta_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        
        """ gray label """
        gray_label = label2gray(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels

def make_batch_bert(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, bert_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, bert_tokenizer))
        
        """ gray label """
        gray_label = label2gray(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)   
    
    batch_input_tokens = padding(batch_input, bert_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels

def make_batch_gpt(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated_gpt(utt, gpt_tokenizer, max_length = 511))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated_gpt(concat_string, gpt_tokenizer, max_length = 511))
        
        """ gray label """
        gray_label = label2gray(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)
    
    batch_input_tokens = padding_gpt(batch_input, gpt_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels


######################### word softmax #########################

def label2gray_ws(emotion, sentiment, gray_type, label_list, sentidict, cos_dict):
    gray_label = []
    if len(label_list) > 3:        
        if gray_type == 'heuristic':
            sim_emotion_list = sentidict[sentiment]
            neu_emotion_list = sentidict['neutral']
            for cand_emo in label_list:
                if cand_emo == emotion:
                    gray_label.append(1)
                elif cand_emo in sim_emotion_list:
                    gray_label.append(0.5)
#                 elif cand_emo in neu_emotion_list:
#                     gray_label.append(0.3)
                else:
                    gray_label.append(0)
        elif gray_type == 'word_softmax':
            for cand_emo in label_list:
                sim = cos_dict[emotion+cand_emo]
                gray_label.append(sim)
        else: # teacher
            print('Error')
            return 
    else:
        if gray_type == 'heuristic':
            for cand_senti in label_list:
                if cand_senti == sentiment:
                    gray_label.append(1)
                elif cand_senti == 'negative' and sentiment == 'positive':
                    gray_label.append(0)
                elif cand_senti == 'negative' and sentiment == 'neutral':
                    gray_label.append(0.5)
                elif cand_senti == 'positive' and sentiment == 'negative':
                    gray_label.append(0)
                elif cand_senti == 'positive' and sentiment == 'neutral':
                    gray_label.append(0.5)
                elif cand_senti == 'neutral' and sentiment == 'positive':
                    gray_label.append(0.5)
                elif cand_senti == 'neutral' and sentiment == 'negative':
                    gray_label.append(0.5)                    
        elif gray_type == 'word_softmax':
            for cand_senti in label_list:
                sim = cos_dict[sentiment+cand_senti]
                gray_label.append(sim)
        else: # teacher
            print('Error')
            return 
    
    if gray_type == 'heuristic':
        gray_label_dist = []
        total_sum = 0
        for gray_label_value in gray_label:
            if gray_label_value > 0:
                total_sum += gray_label_value
        for gray_label_value in gray_label:
            if gray_label_value > 0:
                gray_label_dist.append(gray_label_value/total_sum)
            else:
                gray_label_dist.append(0)
        gray_label_dist = torch.tensor(gray_label_dist)
    elif gray_type == 'word_softmax':
        gray_label_dist = softmax(torch.tensor(gray_label), 0)
    return gray_label_dist.unsqueeze(0) # (1, clsNum)

def make_batch_roberta_ws(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, roberta_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        
        """ gray label """
        gray_label = label2gray_ws(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.cat(batch_labels, 0)
    
    return batch_input_tokens, batch_labels

def make_batch_bert_ws(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, bert_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, bert_tokenizer))
        
        """ gray label """
        gray_label = label2gray_ws(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)   
    
    batch_input_tokens = padding(batch_input, bert_tokenizer)
    batch_labels = torch.cat(batch_labels, 0)
    
    return batch_input_tokens, batch_labels

def make_batch_gpt_ws(sessions):
    batch_input, batch_labels = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated_gpt(utt, gpt_tokenizer, max_length = 511))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated_gpt(concat_string, gpt_tokenizer, max_length = 511))
        
        """ gray label """
        gray_label = label2gray_ws(emotion, sentiment, gray_type, label_list, sentidict, cos_dict)
        batch_labels.append(gray_label)
    
    batch_input_tokens = padding_gpt(batch_input, gpt_tokenizer)
    batch_labels = torch.cat(batch_labels, 0)
    
    return batch_input_tokens, batch_labels