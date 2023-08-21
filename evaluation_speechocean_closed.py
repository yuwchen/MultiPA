import json
import os
import scipy 
from sklearn.metrics import mean_squared_error
import math
import string
import numpy as np
import math
from collections import Counter

def print_result(pred, gt, score_name):
    mse = mean_squared_error(pred, gt)
    corr, _ = scipy.stats.pearsonr(pred, gt)
    spearman, _ = scipy.stats.spearmanr(pred, gt)
    #print('mse:', mse)
    #print('corr:', round(corr,4))
    #print('srcc:', round(spearman,4))
    print(score_name, round(corr,4))


f = open('./speechocean762/resource/scores.json') # path to speechocean score json
data = json.load(f)

test_file = open('./speechocean762/test/wav.scp','r').read().splitlines() # path to speechocean test list
test_data = {}
for line in test_file:
    wavidx = line.split('\t')[0]
    test_data[wavidx] = data[wavidx]
    
def get_prediction(path):
    invalid = 0
    prediction = open(path,'r').read().splitlines()
    result_word = {}
    result_uttr = {}
    for sample in prediction:
        
        parts = sample.split(';')
        wavidx = parts[0].replace('.wav','')
        valid = parts[5].split(':')[1]
        if valid=='F':
            invalid+=1
            accuracy = 1.0
            fluency = 0.0
            prosodic = 0.0
            total = 0.0
            completeness = 0.0
            result_word[wavidx]={}
            result_word[wavidx]['word_accuracy'] = 0
            result_word[wavidx]['word_stress'] = 5
            result_word[wavidx]['word_total'] = 1
            result_word[wavidx]['text'] = ''
        else:
            accuracy = float(parts[1].split(':')[1])
            fluency = float(parts[2].split(':')[1])
            prosodic = float(parts[3].split(':')[1])
            total = float(parts[4].split(':')[1])
            alignment = eval(parts[-1].split(':')[1])
            time_interval  = [float(word[1])-float(word[0]) for word in alignment]
            completeness  = [1 if the_interval > 0.07 else 0 for the_interval in time_interval]
            completeness = sum(completeness)/len(completeness)

            w_a = eval(parts[8].split(':')[1])
            w_s = eval(parts[9].split(':')[1])
            w_t = eval(parts[10].split(':')[1])
            if isinstance(w_a , float):
                w_a = [w_a]
                w_s = [w_s]
                w_t = [w_t]
            w_a = [10 if x > 10 else x for x in w_a]
            w_s = [10 if x > 10 else x for x in w_s]
            w_t = [10 if x > 10 else x for x in w_t]
            result_word[wavidx]={}
            result_word[wavidx]['word_accuracy'] = w_a
            result_word[wavidx]['word_stress'] = w_s
            result_word[wavidx]['word_total'] = w_t
            result_word[wavidx]['text'] = eval(parts[-1].split(':')[1])
            result_word[wavidx]['text'] = [word[-1] for word in result_word[wavidx]['text']]

        result_uttr[wavidx]={}
        result_uttr[wavidx]['accuracy'] = accuracy
        result_uttr[wavidx]['fluency'] = fluency
        result_uttr[wavidx]['prosodic'] = prosodic
        result_uttr[wavidx]['total'] = total
        result_uttr[wavidx]['completeness'] = completeness
    
    #print(invalid)
    return result_word, result_uttr 

def pad_mismatch_sequence(the_gt_w_text, the_pred_w_text, the_pred_w_acc, the_pred_w_stress, the_pred_w_total):
    """
    Sometimes the model will merge consecutive occurrences of the same word. e.g. "seven nine nine one" to "seven nine one"
    In this case, the number of predicted scores won't in line with the number of ground-truth words. 
    Therefore, we duplicate the scores for the merged word. 
    """
    padded_acc = []
    padded_stress = []
    padded_total = []
    asr_w_idx=0

    for gt_word in the_gt_w_text:
        if asr_w_idx>=len(the_pred_w_text):
            padded_acc.append(the_pred_w_acc[asr_w_idx-1])
            padded_stress.append(the_pred_w_stress[asr_w_idx-1])
            padded_total.append(the_pred_w_total[asr_w_idx-1])  
            break
            
        if gt_word == the_pred_w_text[asr_w_idx]:
            padded_acc.append(the_pred_w_acc[asr_w_idx])
            padded_stress.append(the_pred_w_stress[asr_w_idx])
            padded_total.append(the_pred_w_total[asr_w_idx])
            asr_w_idx+=1
        else:
            padded_acc.append(the_pred_w_acc[asr_w_idx-1])
            padded_stress.append(the_pred_w_stress[asr_w_idx-1])
            padded_total.append(the_pred_w_total[asr_w_idx-1])      

    return padded_acc, padded_stress, padded_total 

def calculate_performance(result_word, result_uttr, wav_idx_word, wav_idx_uttr):

    gt_A = []
    gt_F = []
    gt_P = []
    gt_T = []
    gt_C = []

    pred_A = []
    pred_F = []
    pred_P = []
    pred_T = []
    pred_C = []

    for wavidx in wav_idx_uttr:
        gt_A.append(test_data[wavidx]['accuracy'])
        pred_A.append(result_uttr[wavidx]['accuracy'])
        gt_F.append(test_data[wavidx]['fluency'])
        pred_F.append(result_uttr[wavidx]['fluency'])
        gt_P.append(test_data[wavidx]['prosodic'])
        pred_P.append(result_uttr[wavidx]['prosodic'])
        gt_T.append(test_data[wavidx]['total'])
        pred_T.append(result_uttr[wavidx]['total'])
        gt_C.append(test_data[wavidx]['completeness'])
        pred_C.append(result_uttr[wavidx]['completeness'])
    
    print('number of utterance', len(pred_A))
    print_result(pred_A, gt_A, 'sen-accuracy')
    print_result(pred_F, gt_F, 'sen-fluency')
    print_result(pred_P, gt_P, 'sen-prosody')
    print_result(pred_T, gt_T, 'sen-total')
    print_result(pred_C, gt_C,'sen-completeness')

    gt_w_acc = []
    gt_w_stress = []
    gt_w_total = []
    pred_w_acc = []
    pred_w_stress = []
    pred_w_total = []
    count_sen = 0
    for wavidx in wav_idx_word:        
        the_gt_w_acc = []
        the_gt_w_stress = []
        the_gt_w_total = []
        the_gt_w_text = []
        
        for word in test_data[wavidx]['words']:
            the_gt_w_acc.append(int(word['accuracy']))
            the_gt_w_stress.append(int(word['stress']))
            the_gt_w_total.append(int(word['total']))
            the_gt_w_text.append(word['text'].lower())

        the_pred_w_acc = result_word[wavidx]['word_accuracy']
        the_pred_w_stress = result_word[wavidx]['word_stress']
        the_pred_w_total = result_word[wavidx]['word_total']
        
        if len(the_gt_w_text) != len(result_word[wavidx]['text']): #if ground-truth sen and predicted sen not equal in length
            if result_word[wavidx]['text'] == '': # if the ASR cannot recognize the sentence, return the lowest score in the training data 
                gt_len = len(the_gt_w_text)
                the_pred_w_acc = [the_pred_w_acc for _ in range(gt_len)]
                the_pred_w_stress = [the_pred_w_stress for _ in range(gt_len)]
                the_pred_w_total = [the_pred_w_total for _ in range(gt_len)]
            else: # for the case where the forced alignment model merges consecutive occurrences of the same word
                the_pred_w_acc, the_pred_w_stress, the_pred_w_total = pad_mismatch_sequence(the_gt_w_text, result_word[wavidx]['text'], the_pred_w_acc, the_pred_w_stress, the_pred_w_total)

        #assert len(the_gt_w_acc) == len(the_pred_w_acc)
        gt_w_acc.extend(the_gt_w_acc)
        gt_w_stress.extend(the_gt_w_stress)
        gt_w_total.extend(the_gt_w_total)
        pred_w_acc.extend(the_pred_w_acc)
        pred_w_stress.extend(the_pred_w_stress)
        pred_w_total.extend(the_pred_w_total)
        count_sen+=1
               
    #print('number of sentences for word prediction:', count_sen, "# of words:", len(pred_w_acc))
    print_result(pred_w_acc, gt_w_acc, 'word-acc')
    print_result(pred_w_stress, gt_w_stress, 'word-stress')
    print_result(pred_w_total, gt_w_total, 'word-total')



resultA_word, resultA_uttr = get_prediction('./Results/model_assessment_val9_r1_speechocean762_test_gtb.txt')

calculate_performance(resultA_word, resultA_uttr, list(resultA_word.keys()),list(resultA_uttr.keys()))