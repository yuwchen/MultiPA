# MultiPA

This repo is the implementation of the paper:  
MultiPA: a multi-task speech pronunciation assessment system for a closed and open response scenario [[Arxiv]](https://arxiv.org/abs/2308.12490)


 - [Requirement](#Requirement)
 - [Train and evalaute on speechocean762 dataset](#Train-and-evalaute-on-speechocean762-dataset)
 - [Test on you data](#Test-on-you-data)
 - [References](#References)
 - [MultiPA data](#MultiPA-data)
 - [Citation](#Citation)


## Requirement

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
pip install pandas
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
Get the assessment results in a closed response scenario (using ground-truth transcript)
```
python test_closed.py --ckptdir model_assessment
```
The results will be saved in the "model_assessment_speechocean762_test_gtb.txt" with the format:  
{wavname}; A:{uttr-accuracy}; F:{uttr-fluency}; P:{uttr-prosodic}; T:{uttr-total}; Valid:{whether_output_is_valid}; ASR_s:{groud-truth-sentence}; ASR_w:{asr-results}; w_a:{word-accuracy}; w_s:{word-stress}; w_t:{w-total}; alignment:{forced-alignment-result}


Get the assessment results in an open response scenario (using the result of ASRt as a ground-truth transcript)
```
python test_open.py --ckptdir model_assessment
```
The results will be saved in the "model_assessment_speechocean762_test_mb.txt" 


### Step 4. Evaluation
-----
### For closed response scenario

Use "evaluation_speechocean_closed.py". Change the input path of the "get_prediction" function to the path of generated txt file in the Step 3.

Note:  
- Since the whisper might give different recognition results for the same utterance, the performance scores will be slightly different for different runs.
- For utterances that the MultiPA fails to process, the lowest scores in the training data are used. (i.e., accuracy = 1, fluency = 0, prosodic = 0, total = 0, completeness = 0, word_accuracy = 0, word_stress= 5, and word_total = 1.)
- The scores higher than 10 will be clipped to 10. 
- The scores in the paper are the average of five models training with different random seeds.   

Closed response performance (PCC):
| sen-accuracy  | sen-fluency  | sen-prosody   | sen-total   | word-accuracy | word-stress | word-total |
|---------------|--------------|---------------|-------------|---------------|-------------|------------|
| ~0.73         | ~0.79        | ~0.78         | ~0.76       |~0.52           |~0.19       | ~0.53    | 

### Completeness assessment metric

-----
### For open response scenario

Use "evaluation_speechocean_open.py".   
(1) Calculate and save the alignment information of the ground-truth transcript using get_gt_alignment.py. Change the path to dir in line 163.   
(2) Change the input path of the "get_prediction" function to the path of the generated txt file in Step 3.   
Note:  
- The evaluation of the word-level assessment result is different from the closed response scenario because there is a potential mismatch between the ground-truth label and the predicted scores. (the ground-truth labels are aligned with the ground-truth words, whereas the predicted word-level scores are aligned with the ASR-recognized words.)
- Since the whisper might give different recognition results for the same utterance, the performance scores will be slightly different for different runs.

| sen-accuracy  | sen-fluency  | sen-prosody   | sen-total   | word-accuracy | word-stress | word-total |
|---------------|--------------|---------------|-------------|---------------|-------------|------------|
| ~0.70         | ~0.77         | ~0.76        | ~0.73       |~0.42          |~0.24        | ~0.44      |


  
## Test on your data.

```
python api.py --inputdir /path/to/your/wav/dir --ckptdir model_assessment
```
Note: 
- This api works on open response. Please replace "sen_asr_s" with the target sentence if you want to test on closed response. 
- One limitation of MultiPA is its ability to assess long utterances. First, MultiPA might fail to process a long utterance due to the GPU out-off-memory issue. In addition, its  generalization capabilities might be limited because it was trained on utterances shorter than 20 seconds. Therefore, an additional audio segmentation step is recommended when using MultiPA on long utterances. In the api.py, we implement a simple segmentation method based on whisper's results. Specifically, if a wave file is longer than 15 seconds, the model will work on whisper segments and merge (average) the results instead of processing the entire wave file at once.   

Pretrained model:   
Download pre-trained model: [Google Drive](https://drive.google.com/file/d/1Kpm3BeEh6Rh7JZ5tatyHMUMipuo0RYds/view?usp=sharing)  

## References
The Charsiu.py, models.py, processors.py, utils.py in this repo are revised from [Charsiu](https://github.com/lingjzhu/charsiu/tree/main). 
The major change includes:  
(1) return the output embedding (return the probability of all possible phones)    
(2) prevent merging the duration of multiple identical words.     
    (e.g., transcript: very very -> return (s1, e1, very), (s2, e2, very) instead of (s1, e2, very))   
    -> However, in some cases, the model will still return only one alignment result, leading to the mismatch between words in the input sentence and the alignedd words.  

## MultiPA data

Pilot dataset for real-world open response scenario speech assessment. 
Coming soon ..?


## Citation
Please cite our paper if you find this repository useful.
