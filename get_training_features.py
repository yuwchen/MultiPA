import os
import json
import torch
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_assessment import *
from Charsiu import charsiu_forced_aligner
from fairseq.models.roberta import RobertaModel


def get_transcript(df):
      asr_results = {}
      for index, row in df.iterrows():
            the_wavname = row['wavname'].replace('.wav','')
            the_transcript = row['transcript']
            asr_results[the_wavname] = the_transcript
      return asr_results 

def create_dir(outputdir):
      if not os.path.exists(outputdir):
            os.makedirs(outputdir)

def main():


      f = open('./speechocean762/resource/scores.json') # path to speechocean score json
      data = json.load(f)

      train_file = open('./speechocean762/speechocean762_train.txt','r').read().splitlines() 
      train_list = []
      for line in train_file:
            wavname = line.split(';')[0].split('.')[0]
            train_list.append(wavname)
      train_list = list(set(train_list))

      df = pd.read_csv('./whisper_results/speechocean_whisper_all_base_eng.csv')
      #df_m = pd.read_csv('./whisper_results/speechocean_whisper_all_medium_eng.csv')
      #df_s = pd.read_csv('./whisper_results/speechocean_whisper_all_small_eng.csv')
      #df_t = pd.read_csv('./whisper_results/speechocean_whisper_all_tiny_eng.csv')

      whisper_results = get_transcript(df)

      wav_dir = './speechocean762/wav'

      outputdir_pred_words_gt = './feature_base/pred_words_gt'
      outputdir_features_p = './feature_base/features_p'
      outputdir_features_w = './feature_base/features_w'
      outputdir_phone_vector = './feature_base/phone_vector'
      outputdir_gt_word_embed = './feature_base/gt_word_embed'
      outputdir_asr_word_embed  = './feature_base/asr_word_embed'
      outputdir_word_phone_map  = './feature_base/word_phone_map'

      create_dir(outputdir_pred_words_gt)
      create_dir(outputdir_features_p)
      create_dir(outputdir_features_w)
      create_dir(outputdir_phone_vector)
      create_dir(outputdir_gt_word_embed)
      create_dir(outputdir_asr_word_embed)
      create_dir(outputdir_word_phone_map)

      charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
      roberta = RobertaModel.from_pretrained('./fairseq_roberta', checkpoint_file='model.pt')
      roberta.eval()
      
      error_list = open('error_list.txt','w')
      for wavname in tqdm(train_list):
            try:      
                  wavpath = os.path.join(wav_dir , wavname+'.wav')
                  gt_sen_list = []
                  for word in data[wavname]['words']:
                        gt_sen_list.append(word['text'].lower())
                  
                  gt_sen = ' '.join(gt_sen_list)

                  asr_sen = whisper_results[wavname].lower()
                  asr_sen = remove_pun_except_apostrophe(asr_sen)
                  asr_sen = convert_num_to_word(asr_sen)

                  pred_words_gt, features_p, features_w, phone_vector, gt_word_embed, asr_word_embed, word_phone_map = feature_extraction(wavpath, gt_sen, asr_sen, alignment_model=charsiu, word_model=roberta)

                  torch.save(pred_words_gt, os.path.join(outputdir_pred_words_gt, wavname+'.pt'))
                  torch.save(features_p, os.path.join(outputdir_features_p, wavname+'.pt'))
                  torch.save(features_w, os.path.join(outputdir_features_w, wavname+'.pt'))
                  torch.save(phone_vector, os.path.join(outputdir_phone_vector, wavname+'.pt'))
                  torch.save(gt_word_embed, os.path.join(outputdir_gt_word_embed, wavname+'.pt'))
                  torch.save(asr_word_embed, os.path.join(outputdir_asr_word_embed, wavname+'.pt'))
                  torch.save(word_phone_map, os.path.join(outputdir_word_phone_map, wavname+'.pt'))
                  
                  if len(pred_words_gt) != len(gt_sen_list):
                        error_list.write(wavname+'#'+str(pred_words_gt)+'#'+str(gt_sen_list)+'\n')

            except Exception as e:
                 print(e)
                 error_list.write(wavname+'\n')

if __name__ == '__main__':
    main()
