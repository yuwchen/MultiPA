import os
import gc
import json
import time
import argparse
import torch
import copy
import fairseq
import whisper
import numpy as np
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from dataclasses import dataclass
from utils_assessment import *
from model_assessment import PronunciationPredictor
from Charsiu import charsiu_forced_aligner

gc.collect()
torch.cuda.empty_cache()

SSL_OUT_DIM = 768
TEXT_OUT_DIM = 768
SAMPLE_RATE = 16000

def inference_one_seg(sen_asr_s, sen_asr_w, wav):

    sen_asr_s = remove_pun_except_apostrophe(sen_asr_s).lower()
    sen_asr_s = convert_num_to_word(sen_asr_s)

    sen_asr_w = remove_pun_except_apostrophe(sen_asr_w).lower()
    sen_asr_w = convert_num_to_word(sen_asr_w)

    pred_words_gt, features_p, features_w, phonevector, gt_word_embed, asr_word_embed, word_phone_map = feature_extraction(wav.numpy(), sen_asr_s, sen_asr_w, alignment_model=aligment_model, word_model=word_model)

    timesplit =  [[(int(float(word[0])*SAMPLE_RATE), int(float(word[1])*SAMPLE_RATE)) for word in pred_words_gt]]
    word_phone_map = [word_phone_map]

    wav = torch.reshape(wav, (1, -1))
    wav = wav.to(device)
    features_p = torch.from_numpy(features_p).to(device).float().unsqueeze(0)
    features_w = torch.from_numpy(features_w).to(device).float().unsqueeze(0) 
    phonevector = torch.from_numpy(phonevector).to(device).float().unsqueeze(0) 
    gt_word_embed = torch.from_numpy(gt_word_embed).to(device).float().unsqueeze(0) 
    asr_word_embed = torch.from_numpy(asr_word_embed).to(device).float().unsqueeze(0)     

    score_A, score_F, score_P, score_T, w_acc, w_stress, w_total = assessment_model(wav, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit)
    
    torch.cuda.empty_cache()

    return score_A, score_F, score_P, score_T, w_acc, w_stress, w_total, pred_words_gt


def predict_one_file(filepath,  whisper_model_s, whisper_model_w, word_model):
    
    results = {}
    results['wavname'] = filepath.split('/')[-1]    
    with torch.no_grad():

        wav, sr = torchaudio.load(filepath)
        #resample audio recordin to 16000Hz
        if sr!=16000:
            transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = transform(wav)
            sr = SAMPLE_RATE
        
        if wav.shape[0]!=1:
            wav = torch.mean(wav,0)                                                                                          
            
        wav = torch.reshape(wav, (-1, ))

        if wav.shape[0] < SAMPLE_RATE*15: #if input wavfile is less than 15s, process the wavfile at once
            sen_asr_s = get_transcript(wav, whisper_model_s)
            sen_asr_w = get_transcript(wav, whisper_model_w)
            score_A, score_F, score_P, score_T, w_acc, w_stress, w_total, pred_words_gt =  inference_one_seg(sen_asr_s, sen_asr_w, wav)
            score_A = score_A.cpu().detach().numpy()[0]
            score_F = score_F.cpu().detach().numpy()[0]
            score_P = score_P.cpu().detach().numpy()[0]
            score_T = score_T.cpu().detach().numpy()[0] 
            w_a = w_acc.cpu().detach().numpy()[0]
            w_s = w_stress.cpu().detach().numpy()[0]
            w_t = w_total.cpu().detach().numpy()[0]
            results = {}
            pred_words = [word[-1] for word in pred_words_gt]
            results['uttr_acc'] = score_A
            results['uttr_fluency'] = score_F
            results['uttr_prosodic'] = score_P
            results['uttr_total'] = score_T
            results['word_acc'] = w_a
            results['word_stress'] = w_s
            results['word_total'] = w_t
            results['word_text'] = pred_words
            results['transcript_S'] = sen_asr_s  
            results['transcript_W'] = sen_asr_w   

            return results

        else: #if wavfile longer than 15s, do the segmentation to prevent OOM
        
            sen_asr_s_all = get_transcript(wav, whisper_model_s, return_seg=True)
            sen_asr_w_all = ''
            for seg in sen_asr_s_all['segments']:
                sen_asr_s = seg['text']
                start = float(seg['start'])
                end = float(seg['end']) 
                the_wav = wav[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)]
                sen_asr_w = get_transcript(the_wav, whisper_model_w)
                sen_asr_w_all = sen_asr_w_all+' '+sen_asr_w
                the_score_A, the_score_F, the_score_P, the_score_T, the_w_acc, the_w_stress, the_w_total, the_pred_words_gt = inference_one_seg(sen_asr_s, sen_asr_w, the_wav)
                the_pred_words = [word[-1] for word in the_pred_words_gt]
                try:
                    score_A += the_score_A
                    score_F += the_score_F
                    score_P += the_score_P
                    score_T += the_score_T
                    w_acc = torch.cat((w_acc,the_w_acc.squeeze(0)))
                    w_stress = torch.cat((w_stress,the_w_stress.squeeze(0)))
                    w_total = torch.cat((w_acc,the_w_total.squeeze(0)))
                    pred_word.extend(the_pred_words)
                except Exception as e: #first word
                    score_A = the_score_A
                    score_F = the_score_F
                    score_P = the_score_P
                    score_T = the_score_T
                    w_acc = the_w_acc.squeeze(0)
                    w_stress = the_w_stress.squeeze(0)  
                    w_total = the_w_total.squeeze(0)  
                    pred_word = the_pred_words
                
            num_of_seg = len(sen_asr_s_all['segments'])
            results['uttr_acc'] = (score_A/num_of_seg)
            results['uttr_fluency'] = (score_F/num_of_seg)
            results['uttr_prosodic'] = (score_P/num_of_seg)
            results['uttr_total'] = (score_T/num_of_seg)
            results['word_acc'] = w_acc.cpu().detach().numpy()
            results['word_stress'] = w_stress.cpu().detach().numpy()
            results['word_total'] = w_total.cpu().detach().numpy()
            results['word_text'] = pred_word
            results['transcript_S'] = sen_asr_s_all['text']
            results['transcript_W'] = sen_asr_w_all

            return results
            

parser = argparse.ArgumentParser()
parser.add_argument('--fairseq_base_model', type=str, default='./fairseq_hubert/hubert_base_ls960.pt', help='Path to pretrained fairseq hubert model.')
parser.add_argument('--fairseq_roberta', type=str, default='./fairseq_roberta', help='Path to pretrained fairseq roberta.')
parser.add_argument('--inputdir', type=str, help='Path to testing wavfile.')
parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')


args = parser.parse_args()
    
ssl_path = args.fairseq_base_model
roberta_path = args.fairseq_roberta
my_checkpoint_dir = args.ckptdir
file_dir  = args.inputdir

word_model = RobertaModel.from_pretrained(roberta_path, checkpoint_file='model.pt')
word_model.eval()
whisper_model_s = whisper.load_model("medium.en")
whisper_model_w = whisper.load_model("base.en")
aligment_model = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')


print('Loading checkpoint')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

ssl_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
ssl_model = ssl_model[0]

assessment_model = PronunciationPredictor(ssl_model, SSL_OUT_DIM, TEXT_OUT_DIM).to(device)
assessment_model.eval()
assessment_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'PRO'+os.sep+'best')))


filepath_list = get_filepaths(file_dir)

for filepath in filepath_list:
    s = time.time()
    results = predict_one_file(filepath, whisper_model_s, whisper_model_w, word_model)
    print('Process time:', time.time()-s)
    print(results)


