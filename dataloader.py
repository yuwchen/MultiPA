import os
import torch
import numpy as np
import torchaudio
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence


# For MyDatasetW5
ASR_WORD_EMBED_DIR = 'feature_base/asr_word_embed'
GT_WORD_EMBED_DIR = 'feature_base/gt_word_embed'
GT_ALIGNMENT_DIR = 'feature_base/pred_words_gt'
WORD_FEATURE_DIR = 'feature_base/features_w'
PHONE_FEATURE_DIR = 'feature_base/features_p'
PHONEVECTOR_DIR = 'feature_base/phone_vector'
WORD_PHONE_MAP_DIR = 'feature_base/word_phone_map'

SAMPLE_RATE=16000

class MyDataset(Dataset):

    def __init__(self, rootdir, data_list):

        self.A_lookup = {}
        self.F_lookup = {}
        self.P_lookup = {}
        self.T_lookup = {}
        self.w_acc_lookup = {}
        self.w_stress_lookup = {}
        self.w_total_lookup = {}
        self.num_w = {}

        wavfiles = []
        for line in data_list:
            parts = line.split(';')
            wavfile = parts[0]

            A = float(parts[1])
            F = float(parts[2])
            P = float(parts[3])
            T = float(parts[4])
            w_acc = parts[5].split(',')
            w_stress = parts[6].split(',')
            w_total = parts[7].split(',')
            
            w_acc = [float(x) for x in w_acc]
            w_stress = [float(x) for x in w_stress]
            w_total = [float(x) for x in w_total]

            num_of_word = float(parts[8])
            self.A_lookup[wavfile] = A
            self.F_lookup[wavfile] = F
            self.P_lookup[wavfile] = P
            self.T_lookup[wavfile] = T
            self.w_acc_lookup[wavfile] = w_acc
            self.w_stress_lookup[wavfile] = w_stress
            self.w_total_lookup[wavfile] = w_total
            self.num_w[wavfile] = num_of_word

            wavfiles.append(wavfile)

        self.rootdir = rootdir
        self.wavfiles = sorted(wavfiles)
        
    def __getitem__(self, idx):
        wavfile = self.wavfiles[idx]
        wavpath = os.path.join(self.rootdir, wavfile)
        wav = torchaudio.load(wavpath)[0]

        try:
            asr_word_embed = torch.from_numpy(torch.load(os.path.join(ASR_WORD_EMBED_DIR, wavfile.replace('.wav','.pt')))).float()
            gt_word_embed = torch.from_numpy(torch.load(os.path.join(GT_WORD_EMBED_DIR, wavfile.replace('.wav','.pt')))).float()
            gt_alignment = torch.load(os.path.join(GT_ALIGNMENT_DIR, wavfile.replace('.wav','.pt')))
            features_w = torch.from_numpy(torch.load(os.path.join(WORD_FEATURE_DIR, wavfile.replace('.wav','.pt')))).float()
            features_p = torch.from_numpy(torch.load(os.path.join(PHONE_FEATURE_DIR, wavfile.replace('.wav','.pt')))).float()
            phonevector =  torch.from_numpy(torch.load(os.path.join(PHONEVECTOR_DIR, wavfile.replace('.wav','.pt')))).float()
            word_phone_map =  torch.load(os.path.join(WORD_PHONE_MAP_DIR, wavfile.replace('.wav','.pt')))

        except Exception as e:
            print(e, wavfile)
            return None

        num_w = int(len(gt_alignment))

        timesplit = [(int(float(word[0])*SAMPLE_RATE), int(float(word[1])*SAMPLE_RATE)) for word in gt_alignment]
        
        s_A = self.A_lookup[wavfile]
        s_F = self.F_lookup[wavfile]
        s_P = self.P_lookup[wavfile]
        s_T = self.T_lookup[wavfile]

        w_s_acc = torch.tensor(self.w_acc_lookup[wavfile])
        w_s_stress = torch.tensor(self.w_stress_lookup[wavfile])
        w_s_total = torch.tensor(self.w_total_lookup[wavfile])
   

        return wav, s_A, s_F, s_P, s_T, w_s_acc, w_s_stress, w_s_total, timesplit, asr_word_embed, gt_word_embed, features_w, features_p, phonevector, word_phone_map, num_w, wavfile


    def __len__(self):
        return len(self.wavfiles)


    def collate_fn(self, batch):  ## zero padding
        
        batch = list(filter(lambda x: x is not None, batch))

        wav, s_A, s_F, s_P, s_T, w_s_acc, w_s_stress, w_s_total, timesplit, asr_word_embed, gt_word_embed, features_w, features_p, phonevector, word_phone_map, num_w, wavfile = zip(*batch)    
        
        wavs = list(wav)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)

        phonevector = pad_sequence(phonevector, batch_first=True)
        asr_word_embed = pad_sequence(asr_word_embed, batch_first=True)
        gt_word_embed = pad_sequence(gt_word_embed, batch_first=True)
        features_w = pad_sequence(features_w, batch_first=True)
        features_p = pad_sequence(features_p, batch_first=True)

        w_s_acc = pad_sequence(w_s_acc, batch_first=True)
        w_s_stress = pad_sequence(w_s_stress, batch_first=True)
        w_s_total = pad_sequence(w_s_total, batch_first=True)
        s_A  = torch.stack([torch.tensor(x) for x in list(s_A)], dim=0)
        s_F  = torch.stack([torch.tensor(x) for x in list(s_F)], dim=0)
        s_P  = torch.stack([torch.tensor(x) for x in list(s_P)], dim=0)
        s_T  = torch.stack([torch.tensor(x) for x in list(s_T)], dim=0)
        timesplit = list(timesplit)
        word_phone_map = list(word_phone_map)

        return output_wavs, s_A, s_F, s_P, s_T, w_s_acc, w_s_stress, w_s_total, timesplit, asr_word_embed, gt_word_embed, features_w, features_p, phonevector, word_phone_map, num_w, wavfile
    
