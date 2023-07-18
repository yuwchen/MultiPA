# NIPRA

NIPRA: a non-intrusive pronunciation assessment model

## Reproduced the results 


### Data Preparation 
(1) Download the speechocean762 dataset: [Link](https://www.openslr.org/101).   
(2) Put the resource, test, train, and WAVE in the speechocean762 dir.  
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
### Model training
```
python model_assessment.py --outdir model_assessment
```
Note: usually, the validation loss will stop decreasing after 2 epochs.
### Inference
Get the assessment results with the ground-truth transcript
```
python test_gt.py --ckptdir model_assessment
```

Get the assessment results without the ground-truth transcript
```
python test_mb.py --ckptdir model_assessment
```

### Evaluation


## Test on your data.

```
python api.py --inputdir /path/to/your/wav/dir --ckptdir model_assessment
```
Note: to prevent OOM, if a wave file is longer than 15 seconds, the model will work on segments and merge the results instead of processing the entire wave file at once.
