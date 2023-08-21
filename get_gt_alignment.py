import os
import json
import torch
from tqdm import tqdm
from utils_assessment import *
from Charsiu import charsiu_forced_aligner

f = open('./speechocean762/resource/scores.json') # path to speechocean score json
data = json.load(f)


test_file = open('./speechocean762/test/wav.scp','r').read().splitlines() # path to speechocean test list
test_data = {}
for line in test_file:
    wavidx = line.split('\t')[0]
    test_data[wavidx] = data[wavidx]


gt_alignment_dir = './gt_alignment_test'
wav_dir = './speechocean762/wav'
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

for wavidx in tqdm(test_data.keys()):
    
    wavpath = os.path.join(wav_dir , wavidx+'.wav')
    gt_sen_list = []
    for word in data[wavidx]['words']:
        gt_sen_list.append(word['text'].lower())

    gt_sen = ' '.join(gt_sen_list)
    try:
        pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = get_charsiu_alignment(wavpath, gt_sen, charsiu)
        selected_idx = get_match_index(pred_words, words)
        pred_words = np.asarray(pred_words)
        pred_words = pred_words[selected_idx]   
        torch.save(pred_words, os.path.join(gt_alignment_dir, wavidx+'.pt'))
        if len(gt_sen_list)!=len(pred_words):
            print(wavidx)
            print(gt_sen_list)
            print(pred_words)
    except Exception as e:
        print(e)
        print(wavidx)
        print(gt_sen_list)
