# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import json

from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from model import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support

from utils_S_SA import encode_right_truncated, padding
from utils_S_SA import make_batch_roberta, make_batch_bert, make_batch_gpt

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def pdloss(batch_pred_distribution, batch_label_distribution):
    """
    batch_pred_distribution: (batch, clsNum)
    batch_label_distribution: (batch, clsNum)
    """
    batch_log_pred_distribution = torch.log(batch_pred_distribution)
    
    loss_val = 0
    for log_pred_distribution, label_distribution in zip(batch_log_pred_distribution, batch_label_distribution):
        for log_pred_prob, label_prob in zip(log_pred_distribution, label_distribution):
            loss_val -= label_prob*log_pred_prob
    return loss_val

def mod_distribution(batch_labels, pred_teacher_distribution):
    mod_teacher_distribution = []
    for max_ind, pred_teacher_dist in zip(batch_labels, pred_teacher_distribution):
        pred_max_ind = pred_teacher_dist.argmax(0)
        
#         if pred_max_ind == max_ind:
        if pred_teacher_dist[max_ind] >= 0.5:
            mod_teacher_dist = pred_teacher_dist
        else:
            remain_prob_sum = 1-pred_teacher_dist[max_ind]
            mod_teacher_dist = torch.zeros(pred_teacher_dist.shape)
            mod_teacher_dist[max_ind] = 0.5
            for i, prob in enumerate(pred_teacher_dist):
                if i != max_ind:
                    mod_teacher_dist[i] = 0.5*pred_teacher_dist[i]/remain_prob_sum
        mod_teacher_distribution.append(mod_teacher_dist.unsqueeze(0).type('torch.cuda.FloatTensor'))
    mod_teacher_distribution = torch.cat(mod_teacher_distribution, 0)
    return mod_teacher_distribution
    
## finetune RoBETa-large
def main():
    """ word embedding """
    with open('word_emb/emotion.json', "r") as json_file:
        word_emb = json.load(json_file)
        
    """Dataset Loading"""
    batch_size = args.batch
    dataset = args.dataset
    dataclass = args.cls
    sample = args.sample
    model_type = args.pretrained
    gray_type = args.gray
    w1 = args.weight1
    w2 = args.weight2
    
    dataType = 'multi'
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = './dataset/MELD/'+dataType+'/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = './dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = './dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
    elif dataset == 'dailydialog':
        data_path = './dataset/dailydialog/'
        DATA_loader = DD_loader
        
    if model_type == 'roberta-large':
        make_batch = make_batch_roberta
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert
    else:
        make_batch = make_batch_gpt        
        
    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'
            
    train_dataset = DATA_loader(train_path, dataclass, gray_type, word_emb)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    train_sample_num = int(len(train_dataset)*sample)
    
    dev_dataset = DATA_loader(dev_path, dataclass, gray_type, word_emb)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    test_dataset = DATA_loader(test_path, dataclass, gray_type, word_emb)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    """logging and path"""
    save_path = os.path.join(dataset+'_models', model_type, dataclass, gray_type, str(w1)+'_'+str(w2))
    
    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    if 'gpt2' in model_type:
        last = True
    else:
        last = False
        
    print('DataClass: ', dataclass, '!!!') # emotion    
    clsNum = len(train_dataset.labelList)
    model = ERC_model(model_type, clsNum, last)
    model = model.cuda()    
    model.train() 
    
    """Teacher model Loading"""
    teacher_path = os.path.join('../self', dataset+'_models', model_type, dataclass)
    print("###Teacher Path### ", teacher_path)
    teacher_model = ERC_model(model_type, clsNum, last)
    modelfile = os.path.join(teacher_path, 'model.bin')
    teacher_model.load_state_dict(torch.load(modelfile))
    teacher_model = teacher_model.cuda()    
    teacher_model.eval()     
    
    """Training Setting"""        
    training_epochs = args.epoch
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0    
    best_epoch = 0
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(train_dataloader):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            """Prediction"""
            batch_input_tokens, batch_labels = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda() # type('torch.cuda.FloatTensor')
            
            pred_logits = model(batch_input_tokens)
            pred_teacher_logits = teacher_model(batch_input_tokens)

            """Loss calculation & training"""
            loss_val = 0
            loss_val += w1*CELoss(pred_logits, batch_labels)
            
            pred_distribution = softmax(pred_logits, 1)
            pred_teacher_distribution = softmax(pred_teacher_logits, 1)
            
            if gray_type == 'teacher_post':
                mod_teacher_distribution = mod_distribution(batch_labels, pred_teacher_distribution)
                loss_val += w2*pdloss(pred_distribution, mod_teacher_distribution)
            else:
                loss_val += w2*pdloss(pred_distribution, pred_teacher_distribution)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        model.eval()
        if dataset == 'dailydialog': # micro & macro
            dev_prek, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
            
            dev_fscore = dev_fbeta_macro+dev_fbeta_micro

            """Best Score & Model Save"""
            if dev_fscore > best_dev_fscore_macro + best_dev_fscore_micro:
                best_dev_fscore_macro = dev_fbeta_macro                
                best_dev_fscore_micro = dev_fbeta_micro
                
                test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
                test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
                test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x                
                
                best_epoch = epoch
                _SaveModel(model, save_path)                
        else: # weight
            dev_prek, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            """Best Score & Model Save"""
            if dev_fbeta > best_dev_fscore:
                best_dev_fscore = dev_fbeta
                
                test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
                test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
                
                best_epoch = epoch
                _SaveModel(model, save_path)
        
        logger.info('Epoch: {}'.format(epoch))
        if dataset == 'dailydialog': # micro & macro
            logger.info('Devleopment ## precision: {}, macro-fscore: {}, micro-fscore: {}'.format(dev_prek, dev_fbeta_macro, dev_fbeta_micro))
            logger.info('')
        else:
            logger.info('Devleopment ## precision: {}, precision: {}, recall: {}, fscore: {}'.format(dev_prek, dev_pre, dev_rec, dev_fbeta))
            logger.info('')
        
    if dataset == 'dailydialog': # micro & macro
        logger.info('Final Fscore ## test-precision: {}, test-macro: {}, test-micro: {}, test_epoch: {}'.format(test_prek, test_fbeta_macro, test_fbeta_micro, best_epoch))
    else:
        logger.info('Final Fscore ## test-precision: {}, test-fscore: {}, test_epoch: {}'.format(test_prek, test_fbeta, best_epoch))        
    
def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    p1num, p2num, p3num = 0, 0, 0    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_labels = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = indices[0] # pred_logits.argmax(1).item()            
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
                
            """Calculation precision"""
            if true_label in indices[:1]:
                p1num += 1
            if true_label in indices[:2]:
                p2num += 1/2
            if true_label in indices[:3]:
                p3num += 1/3
            
        p1 = round(p1num/len(dataloader)*100, 2)
        p2 = round(p2num/len(dataloader)*100, 2)
        p3 = round(p3num/len(dataloader)*100, 2)
    return [p1, p2, p3], pred_list, label_list
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
    parser.add_argument( "--weight1", type=float, help = "weighted loss for original", default = 1.0) # 
    parser.add_argument( "--weight2", type=float, help = "weighted loss for grayscale", default = 1.0) # 
    parser.add_argument( "--gray", help = 'teacher or teacher_post', default = 'teacher')

    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog', default = 'MELD')
    
    parser.add_argument( "--pretrained", help = 'roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    