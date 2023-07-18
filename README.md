# NIPRA

NIPRA: a non-intrusive pronunciation assessment model

### Requirements

#### Download pre-trained model
(1) Download [HuBERT Base (~95M params)](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md), and put the hubert_base_ls960.pt in fairseq_hubert dir.   
(2) Download [roberta.base model](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md), and put the model.pt and dict.txt in fairseq_roberta dir.

## Reproduced the results 

### Step 1. Data Preparation 
(1) Download the speechocean762 dataset: [Link](https://www.openslr.org/101).   
(2) Put the resource, test, train, and WAVE in the speechocean762 dir.    
  -   i.e., merge the speechocean762 dir in this repo and the download speechocean762 dir. 
    
(3) Convert .WAV file to .wav file using:
```
python convert_wavfile.py
```
(4) Generate the training and testing list using:
```
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

Get the assessment results without the ground-truth transcript
```
python test_mb.py --ckptdir model_assessment
```

### Step 4. Evaluation


## Test on your data.

```
python api.py --inputdir /path/to/your/wav/dir --ckptdir model_assessment
```
Note: to prevent OOM, if a wave file is longer than 15 seconds, the model will work on segments and merge the results instead of processing the entire wave file at once.

## References:
The Charsiu.py, models.py, processors.py, utils.py in this repo are revised from [Charsiu](https://github.com/lingjzhu/charsiu/tree/main). 
The major change includes:
(1) return the output embedding (return the probability of all possible phones)
(2) prevent merging the duration of multiple identical words.   
    (e.g., very very -> return (s1, e1, very), (s2, e2, very) instead of (s1, e2, very))
