import torch

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

roberta_tokenizer = RobertaTokenizer.from_pretrained('/data/project/rw/rung/model/roberta-large/')
bert_tokenizer = BertTokenizer.from_pretrained('/data/project/rw/rung/model/bert-large-uncased/')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('/data/project/rw/rung/model/gpt2-large/')
gpt_tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})

# condition_token = ['<s1>', '<s2>', '<s3>'] # 최대 3명
# special_tokens = {'additional_special_tokens': condition_token}
# roberta_tokenizer.add_special_tokens(special_tokens)
# bert_tokenizer.add_special_tokens(special_tokens)
# gpt_tokenizer.add_special_tokens(special_tokens)

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

def make_batch_roberta(sessions):
    batch_input, batch_labels = [], []
    batch_all_input, batch_cls_positions = [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        gray_type = session[2]
        sentidict = session[3]
        cos_dict = session[4]
        
        context_speaker, context, emotion, sentiment, all_context_speaker, all_context = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        turn = all_context.index(context[-1])
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, roberta_tokenizer))
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))                
                
        all_inputString = ""
        for now_turn, (speaker, utt) in enumerate(zip(all_context_speaker, all_context)):
            if now_turn > turn+2:
                break
            if now_turn == turn:
                all_inputString += roberta_tokenizer.cls_token + " "
            all_inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...            
            all_inputString += utt + " "                 
        all_concat_string = all_inputString.strip()
        batch_all_input.append(encode_right_truncated(all_concat_string, roberta_tokenizer))        
        
        """ true label """
        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    batch_all_input_tokens = padding(batch_all_input, roberta_tokenizer)
    
    for input_tokens in batch_all_input_tokens.tolist():
        cls_position = input_tokens.index(roberta_tokenizer.cls_token_id)
        batch_cls_positions.append(cls_position)    
    
    return batch_input_tokens, batch_labels, batch_all_input_tokens, batch_cls_positions

# def make_batch_bert(sessions):
#     batch_input, batch_labels = [], []
#     for session in sessions:
#         data = session[0]
#         label_list = session[1]
#         gray_type = session[2]
#         sentidict = session[3]
#         cos_dict = session[4]
        
#         context_speaker, context, emotion, sentiment = data
#         now_speaker = context_speaker[-1]
#         speaker_utt_list = []
        
#         inputString = ""
#         for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
#             inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
#             inputString += utt + " "
            
#             if turn<len(context_speaker)-1 and speaker == now_speaker:
#                 speaker_utt_list.append(encode_right_truncated(utt, bert_tokenizer))
        
#         concat_string = inputString.strip()
#         batch_input.append(encode_right_truncated(concat_string, bert_tokenizer))
        
#         """ true label """
#         if len(label_list) > 3:
#             label_ind = label_list.index(emotion)
#         else:
#             label_ind = label_list.index(sentiment)
#         batch_labels.append(label_ind)
    
#     batch_input_tokens = padding(batch_input, bert_tokenizer)
#     batch_labels = torch.tensor(batch_labels)    
    
#     return batch_input_tokens, batch_labels

# def make_batch_gpt(sessions):
#     batch_input, batch_labels = [], []
#     for session in sessions:
#         data = session[0]
#         label_list = session[1]
#         gray_type = session[2]
#         sentidict = session[3]
#         cos_dict = session[4]
        
#         context_speaker, context, emotion, sentiment = data
#         now_speaker = context_speaker[-1]
#         speaker_utt_list = []
        
#         inputString = ""
#         for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
#             inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
#             inputString += utt + " "
            
#             if turn<len(context_speaker)-1 and speaker == now_speaker:
#                 speaker_utt_list.append(encode_right_truncated_gpt(utt, gpt_tokenizer, max_length = 511))
        
#         concat_string = inputString.strip()
#         batch_input.append(encode_right_truncated_gpt(concat_string, gpt_tokenizer, max_length = 511))
        
#         """ true label """
#         if len(label_list) > 3:
#             label_ind = label_list.index(emotion)
#         else:
#             label_ind = label_list.index(sentiment)
#         batch_labels.append(label_ind)
    
#     batch_input_tokens = padding_gpt(batch_input, gpt_tokenizer)
#     batch_labels = torch.tensor(batch_labels)    
    
#     return batch_input_tokens, batch_labels