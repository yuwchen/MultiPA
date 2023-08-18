# MultiPA

This repo is the implementation of the paper:
MultiPA: a multi-task speech pronunciation assessment system for a closed and open response scenario
[Arxiv]

### Installation

```
conda create -n MultiPA python=3.9
conda activate MultiPA
pip install fairseq
pip install soundfile
pip install -U openai-whisper
pip install transformers
pip install num2words
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.3.0
pip install levenshtein
pip install nltk
pip install praatio
pip install g2pM
pip install librosa
pip install g2p-en
```
Note: spacy needs to be 2.x version


#### Download pre-trained model
(1) Download [HuBERT Base (~95M params)](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md), and put the hubert_base_ls960.pt in fairseq_hubert dir.   
(2) Download [roberta.base model](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md), and put the model.pt and dict.txt in fairseq_roberta dir.

## Reproduce the results 

### Step 1. Data Preparation 
(1) Download the speechocean762 dataset: [Link](https://www.openslr.org/101).   
(2) Put the resource, test, train, and WAVE in the speechocean762 dir.    
  -   i.e., merge the speechocean762 dir in this repo and the download speechocean762 dir. 
    
(3) Convert .WAV file to .wav file using:
```
cd speechocean762
python convert_wavfile.py
```
(4) Generate the training and testing list using:
```
cd speechocean762
python create_training_and_testing_list.py
```
(5) Obtain the whisper ASR result. 
```
python whisper_asr_all.py
```
(6) Generate training features
```
python get_training_features.py
```

### Step 2. Model training
```
python model_assessment.py --outdir model_assessment
```
Note: usually, the validation loss will stop decreasing after 2 epochs.

### Step 3. Inference
Get the assessment results with the ground-truth transcript
```
python test_gt.py --ckptdir model_assessment
```
The results will be saved in the "model_assessment_speechocean762_test_gtb.txt" with the format:  
{wavname}; A:{uttr-accuracy}; F:{uttr-fluency}; P:{uttr-prosodic}; T:{uttr-total}; Valid:{whether_output_is_valid}; ASR_s:{groud-truth-sentence}; ASR_w:{asr-results}; w_a:{word-accuracy}; w_s:{word-stress}; w_t:{w-total}; alignment:{forced-alignment-result}


### Step 4. Evaluation

Use "evaluation_speechocean.py". Change the input path of the "get_prediction" function to the path of generated txt file in the Step 3.

Note:  
- Since the whisper might give different recognition results for the same utterance, the performance scores will be slightly different for different runs.
- For utterances that the MultiPA fails to process, the lowest scores in the training data are used. (i.e., accuracy = 1, fluency = 0, prosodic = 0, total = 0, completeness = 0, word_accuracy = 0, word_stress= 5, and word_total = 1.)  
- The scores in the paper are the average of five models training with different random seeds.   

Performance (PCC):
| sen-accuracy | sen-fluency   | sen-prosody   | sen-total  | word-accuracy | word-stress | word-total |
|---------------|--------------|---------------|-------------|---------------|-------------|------------|
| ~0.7330       | ~0.7971      | ~0.7871       | ~0.7608     |~0.5220        |~0.1999      | ~0.5314    | 


## Test on your data.

```
python api.py --inputdir /path/to/your/wav/dir --ckptdir model_assessment
```
Note: to prevent OOM, if a wave file is longer than 15 seconds, the model will work on segments and merge the results instead of processing the entire wave file at once.

Pretrained model:   
Download pre-trained model: [Google Drive](https://drive.google.com/file/d/1Kpm3BeEh6Rh7JZ5tatyHMUMipuo0RYds/view?usp=sharing)  

## References
The Charsiu.py, models.py, processors.py, utils.py in this repo are revised from [Charsiu](https://github.com/lingjzhu/charsiu/tree/main). 
The major change includes:  
(1) return the output embedding (return the probability of all possible phones)   
(2) prevent merging the duration of multiple identical words.    
    (e.g., transcript: very very -> return (s1, e1, very), (s2, e2, very) instead of (s1, e2, very))  
    -> However, in some cases, the model will still return only one alignment result, leading to the mismatch between words in the input sentence and the alignmened words. 

## Citation
