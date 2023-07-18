import os
import gc
import json
import argparse
import torch
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, default='./fairseq_hubert/hubert_base_ls960.pt', help='Path to pretrained fairseq hubert model.')
    parser.add_argument('--fairseq_roberta', type=str, default='./fairseq_roberta', help='Path to pretrained fairseq roberta.')
    parser.add_argument('--speechocean_gt', type=str, default='./speechocean762/resource/scores.json', help='Path to speechocean scores.json')
    parser.add_argument('--datadir', default='./speechocean762/wav', type=str, help='Path of your DATA/ directory')
    parser.add_argument('--datalist', default='./speechocean762/speechocean762_test.txt', type=str, help='')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')


    args = parser.parse_args()
    
    ssl_path = args.fairseq_base_model
    roberta_path = args.fairseq_roberta
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    datalist = args.datalist

    f = open(args.speechocean_gt)
    gt_data = json.load(f)

    word_model = RobertaModel.from_pretrained(roberta_path, checkpoint_file='model.pt')
    word_model.eval()
    whisper_model_w = whisper.load_model("base.en")

    aligment_model = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

    SSL_OUT_DIM = 768
    TEXT_OUT_DIM = 768
    SAMPLE_RATE = 16000
    
    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    ssl_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
    ssl_model = ssl_model[0]
   
    assessment_model = PronunciationPredictor(ssl_model, SSL_OUT_DIM, TEXT_OUT_DIM).to(device)
    assessment_model.eval()
    assessment_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'PRO'+os.sep+'best')))

    print('Loading data')
    validset = open(datalist,'r').read().splitlines()
    outfile = my_checkpoint_dir.split("/")[-1]+'_'+datalist.split('/')[-1].replace('.txt','_gtb.txt')

    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    prediction = open(os.path.join(output_dir, outfile), 'w')
    
    print('Starting prediction')
    for filename in tqdm(validset):
        
        with torch.no_grad():
            if datalist is not None:
                filepath = os.path.join(datadir, filename)
            else:
                filepath=filename
            wav, sr = torchaudio.load(filepath)
            #resample audio recordin to 16000Hz
            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = transform(wav)
                sr = SAMPLE_RATE
        
            sen_asr_s = []
            for word in gt_data[filename.replace('.wav','')]['words']:
                sen_asr_s.append(word['text'].lower())
            sen_asr_s = ' '.join(sen_asr_s)
            
            wav  = torch.reshape(wav, (-1,))
            
            sen_asr_w = remove_pun_except_apostrophe(get_transcript(wav, whisper_model_w)).lower()
            sen_asr_w = convert_num_to_word(sen_asr_w)

            try:
                
                pred_words_gt, features_p, features_w, phonevector, gt_word_embed, asr_word_embed, word_phone_map = feature_extraction(wav.numpy(), sen_asr_s, sen_asr_w, alignment_model=aligment_model, word_model=word_model)

                timesplit =  [[(int(float(word[0])*SAMPLE_RATE), int(float(word[1])*SAMPLE_RATE)) for word in pred_words_gt]] 
                word_phone_map = [word_phone_map]

                features_p = torch.from_numpy(features_p).to(device).float().unsqueeze(0)
                features_w = torch.from_numpy(features_w).to(device).float().unsqueeze(0) 
                phonevector = torch.from_numpy(phonevector).to(device).float().unsqueeze(0) 
                gt_word_embed = torch.from_numpy(gt_word_embed).to(device).float().unsqueeze(0) 
                asr_word_embed = torch.from_numpy(asr_word_embed).to(device).float().unsqueeze(0) 
                wav = wav.to(device).unsqueeze(0)

                score_A, score_F, score_P, score_T, w_acc, w_stress, w_total = assessment_model(wav, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit)
                score_A = score_A.cpu().detach().numpy()[0]
                score_F = score_F.cpu().detach().numpy()[0]
                score_P = score_P.cpu().detach().numpy()[0]
                score_T = score_T.cpu().detach().numpy()[0] 
                w_a = w_acc.cpu().detach().numpy()[0]
                w_s = w_stress.cpu().detach().numpy()[0]
                w_t = w_total.cpu().detach().numpy()[0]
                
                w_a = ','.join([str(num) for num in w_a])
                w_s = ','.join([str(num) for num in w_s])
                w_t = ','.join([str(num) for num in w_t])

                valid = 'T'
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; alignment:{}".format(filename, score_A, score_F, score_P, score_T, valid, sen_asr_s, sen_asr_w, w_a, w_s, w_t, pred_words_gt.tolist())
                print(output)
                prediction.write(output+'\n')

            except Exception as e:
                print(e)
                valid = 'F'
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; alignment:{}".format(filename, '', '', '', '', valid, '', '', '', '', '', '')
                prediction.write(output+'\n')
                continue
               

            torch.cuda.empty_cache()

            
 

if __name__ == '__main__':
    main()
