import os
import argparse
import fairseq
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import MyDataset


random.seed(1984)


class PronunciationPredictor(nn.Module):
    
    def __init__(self, ssl_model, ssl_out_dim, text_out_dim):
        super(PronunciationPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim  # size of HuBERT embedding
        self.text_out_dim = text_out_dim # size of roberta embedding
        self.w_features_size = 10
        self.p_align_size = 9
        self.p_pred_size = 42
        self.phone_vector = 71*2

        hidden_word = self.ssl_features+self.text_out_dim*2+self.phone_vector+self.w_features_size+self.p_align_size+self.p_pred_size*2+1
        hidden_sen = 768+hidden_word

        self.output_accuracy = nn.Linear(hidden_sen, 1) 
        self.output_fluency = nn.Linear(hidden_sen, 1) 
        self.output_prosodic = nn.Linear(hidden_sen, 1) 
        self.output_total = nn.Linear(hidden_sen, 1)

        self.fusion_layer = nn.TransformerEncoderLayer(d_model=hidden_word, nhead=30)
        
        self.fusion_layer_word = nn.TransformerEncoderLayer(d_model=hidden_word, nhead=30)
        self.w_feature_layer = nn.TransformerEncoderLayer(d_model=self.w_features_size, nhead=5)
        self.p_feature_layer_align = nn.TransformerEncoderLayer(d_model=self.p_align_size, nhead=3)
        self.p_feature_layer_pred_gt = nn.TransformerEncoderLayer(d_model=self.p_pred_size, nhead=7)
        self.p_feature_layer_pred_asr = nn.TransformerEncoderLayer(d_model=self.p_pred_size, nhead=7)
        self.p_feature_layer_1 = nn.TransformerEncoderLayer(d_model=self.p_align_size+self.p_pred_size*2, nhead=3)
        self.p_feature_layer_2 = nn.TransformerEncoderLayer(d_model=self.p_align_size+self.p_pred_size*2, nhead=3)
        self.phonevector_layer = nn.TransformerEncoderLayer(d_model=self.phone_vector, nhead=2)

        self.word_acc = nn.Conv1d(hidden_word, 1, kernel_size=1)
        self.word_stress = nn.Conv1d(hidden_word, 1, kernel_size=1)
        self.word_total = nn.Conv1d(hidden_word, 1, kernel_size=1)

    def forward(self, wav, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit):
        
        res = self.ssl_model(wav, mask=False, features_only=True)
        wav_embedding_raw = res['x']
        
        ### align word-level features to the wavform
        batch_size  = gt_word_embed.shape[0]
        wav_aligned = torch.zeros((gt_word_embed.shape[0], gt_word_embed.shape[1], self.ssl_features)).cuda()
        for b_idx in range(batch_size):
            for w_idx in range(len(timesplit[b_idx])):
                start_point = timesplit[b_idx][w_idx][0] // 320
                end_point = timesplit[b_idx][w_idx][1] // 320
                if (end_point - start_point)==0:  # avoid predict nan because of no aligned wav segment                 
                    the_word = wav_embedding_raw[b_idx, start_point:start_point+1, :]
                else:
                    the_word = wav_embedding_raw[b_idx, start_point:end_point, :]
                aligned_wav_embed = the_word.mean(dim=0)
                wav_aligned[b_idx, w_idx, :] = aligned_wav_embed


        features_w = self.w_feature_layer(features_w)
        features_p = self.p_feature_layer_1(features_p)
        features_p[:,:,:self.p_align_size] = self.p_feature_layer_align(features_p[:,:,:self.p_align_size])
        features_p[:,:,self.p_align_size:self.p_align_size+self.p_pred_size] = self.p_feature_layer_pred_gt(features_p[:,:,self.p_align_size:self.p_align_size+self.p_pred_size])
        features_p[:,:,self.p_align_size+self.p_pred_size:] = self.p_feature_layer_pred_asr(features_p[:,:,self.p_align_size+self.p_pred_size:])

        # align phone-level features to word-level features
        features_p_aligned = torch.zeros((gt_word_embed.shape[0], gt_word_embed.shape[1], self.p_align_size+self.p_pred_size*2)).cuda()
        for b_idx in range(batch_size):
            for w_idx, p_list in enumerate(word_phone_map[b_idx]):
                features_p_aligned[b_idx, w_idx, :] = features_p[b_idx,p_list,:].mean(dim=0)

        features_p_aligned = self.p_feature_layer_2(features_p_aligned)
        phonevector = self.phonevector_layer(phonevector)
        fusion = torch.cat([wav_aligned, gt_word_embed, asr_word_embed, features_w, features_p_aligned, phonevector], dim=2)  
        fusion = F.pad(fusion, (0, 1), mode='constant') # expand one dimension because original feature size is a prime number

        fusion_word = self.fusion_layer_word(fusion)
        fusion_word = fusion_word.transpose(1, 2) 
        output_w_acc = self.word_acc(fusion_word)
        output_w_stress = self.word_stress(fusion_word)
        output_w_total = self.word_total(fusion_word)
        output_w_acc = output_w_acc.transpose(1,2).squeeze(2) 
        output_w_stress = output_w_stress.transpose(1,2).squeeze(2)
        output_w_total = output_w_total.transpose(1,2).squeeze(2)
        

        fusion = self.fusion_layer(fusion) 
        uttr_word = torch.mean(fusion, 1)
        wav_embedding = torch.mean(wav_embedding_raw, 1)
        uttr = torch.cat([wav_embedding, uttr_word], dim=1)
        output_A = self.output_accuracy(uttr)
        output_F = self.output_fluency(uttr)
        output_P = self.output_prosodic(uttr)
        output_T = self.output_total(uttr)
        

        return output_A.squeeze(1), output_F.squeeze(1), output_P.squeeze(1), output_T.squeeze(1), output_w_acc, output_w_stress, output_w_total 
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./speechocean762/wav',  type=str, help='Path to root data directory')
    parser.add_argument('--txtfiledir', default='./speechocean762',  type=str, help='Path to training txt directory')
    parser.add_argument('--fairseq_base_model', default='./fairseq_hubert/hubert_base_ls960.pt', type=str, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to the checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='model_assessment_r2', help='Output directory for trained checkpoints')

    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    txtfiledir = args.txtfiledir
    my_checkpoint_dir = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir,'PRO'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, '')
    input_list = open(os.path.join(txtfiledir, 'speechocean762_train.txt'),'r').read().splitlines()
    random.shuffle(input_list)

    trainlist = input_list[:int(len(input_list)*0.9)]
    validlist = input_list[int(len(input_list)*0.9):]

    SSL_OUT_DIM = 768
    TEXT_OUT_DIM = 768

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=2, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    net = PronunciationPredictor(ssl_model, SSL_OUT_DIM, TEXT_OUT_DIM)
    net = net.to(device)

    if my_checkpoint_dir != None:  
        net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'PRO','best')))

    criterion = nn.MSELoss() 
    optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.7)

    PREV_VAL_LOSS=9999999999
    orig_patience=2
    patience=orig_patience
    
    for epoch in range(1,10):
        STEPS=0
        net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            wav, s_A, s_F, s_P, s_T, w_s_acc, w_s_stress, w_s_total, timesplit, asr_word_embed, gt_word_embed, features_w, features_p, phonevector, word_phone_map, _, wavname = data
            wav = wav.to(device)
            s_A = s_A.to(device)
            s_F = s_F.to(device)
            s_P = s_P.to(device)
            s_T = s_T.to(device)
            w_s_acc = w_s_acc.to(device)
            w_s_stress = w_s_stress.to(device)
            w_s_total = w_s_total.to(device)
            asr_word_embed = asr_word_embed.to(device)
            gt_word_embed = gt_word_embed.to(device)
            features_w =  features_w.to(device)
            features_p =  features_p.to(device)
            phonevector = phonevector.to(device)

            wav_input = wav.squeeze(1)  
            optimizer.zero_grad()
            output_A, output_F, output_P, output_T, output_w_acc, output_w_stress, output_w_total = net(wav_input, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit)
            if output_w_acc.shape[1]!=w_s_acc.shape[1]: 
                continue
            loss_A = criterion(output_A, s_A)
            loss_F = criterion(output_F, s_F)
            loss_P = criterion(output_P, s_P)
            loss_T = criterion(output_T, s_T)
            loss_wa = criterion(output_w_acc, w_s_acc)
            loss_ws = criterion(output_w_stress, w_s_stress)
            loss_wt = criterion(output_w_total, w_s_total)
            loss = loss_A  + loss_F + loss_P + loss_T + loss_wa + loss_ws + loss_wt

            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()

        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))


        ## validation
        VALSTEPS=0
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 

        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1

            wav, s_A, s_F, s_P, s_T, w_s_acc, w_s_stress, w_s_total, timesplit, asr_word_embed, gt_word_embed, features_w, features_p, phonevector, word_phone_map,  _, _ = data
            wav = wav.to(device)
            s_A = s_A.to(device)
            s_F = s_F.to(device)
            s_P = s_P.to(device)
            s_T = s_T.to(device)
            w_s_acc = w_s_acc.to(device)
            w_s_stress = w_s_stress.to(device)
            w_s_total = w_s_total.to(device)
            asr_word_embed = asr_word_embed.to(device)
            gt_word_embed = gt_word_embed.to(device)
            features_w =  features_w.to(device)
            features_p =  features_p.to(device)
            phonevector = phonevector.to(device)

            wav_input = wav.squeeze(1)  

            with torch.no_grad(): 
                output_A, output_F, output_P, output_T, output_w_acc, output_w_stress, output_w_total = net(wav_input, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit)
                if output_w_acc.shape[1]!=w_s_acc.shape[1]: 
                    continue                
                loss_A = criterion(output_A, s_A)
                loss_F = criterion(output_F, s_F)
                loss_P = criterion(output_P, s_P)
                loss_T = criterion(output_T, s_T)
                loss_wa = criterion(output_w_acc, w_s_acc)
                loss_ws = criterion(output_w_stress, w_s_stress)
                loss_wt = criterion(output_w_total, w_s_total)
                loss = loss_A  + loss_F + loss_P + loss_T + loss_wa + loss_ws + loss_wt

                epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(net.state_dict(), os.path.join(ckptdir,'PRO','best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training of Pronunciation Assessment Model')

if __name__ == '__main__':
    main()
