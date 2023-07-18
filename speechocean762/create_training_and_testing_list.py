import json
import os
import numpy as np

def get_filepaths(directory):
      file_paths = []  
      for root, directories, files in os.walk(directory):
            for filename in files:
                  # Join the two strings in order to form the full filepath.
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths  

f = open('./resource/scores-detail.json')
data = json.load(f)


train_file = open('./train/wav.scp','r').read().splitlines()
test_file = open('./test/wav.scp','r').read().splitlines()

train_out = open('./speechocean762_train.txt','w')
test_out = open('./speechocean762_test.txt','w')

for line in train_file:
      wavidx = line.split('\t')[0]
      the_data = data[wavidx]
      accuracy = the_data['accuracy']
      completeness = the_data['completeness']
      fluency = the_data['fluency']
      prosodic = the_data['prosodic']
      total = the_data['total']

      for idx in range(5): 
            W_acc_list = []
            W_stress_list = []
            W_total_list = []
            sen_length = len(the_data['words'])
            for w_idx in range(sen_length):
                  w_acc = str(the_data['words'][w_idx]['accuracy'][idx])
                  w_stress = str(the_data['words'][w_idx]['stress'][idx])
                  w_total = str(the_data['words'][w_idx]['total'][idx])
                  W_acc_list.append(w_acc)
                  W_stress_list.append(w_stress)
                  W_total_list.append(w_total)

            word_acc = ','.join(W_acc_list)
            word_stress = ','.join(W_stress_list)
            word_total = ','.join(W_total_list)
            raw = '{}.wav;{};{};{};{};{};{};{};{}\n'.format(wavidx, accuracy[idx], fluency[idx], prosodic[idx], total[idx], word_acc, word_stress, word_total, sen_length)    
            train_out.write(raw)        


for line in test_file:
      wavidx = line.split('\t')[0] 
      test_out.write(wavidx+'.wav\n')  