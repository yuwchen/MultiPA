import os
import nltk
import string
import num2words
import numpy as np
from dataclasses import dataclass
from Levenshtein import ratio
from nltk.corpus import cmudict
from difflib import SequenceMatcher


def fit_one_hot(inputlist):
    mapping = {}
    for i in range(len(inputlist)):
        mapping[inputlist[i]]=i
    return mapping

"""
load the cumdict and convert it to dict for shorter processing time
"""
nltk.download('cmudict')
all_phonemes = set()
entries = cmudict.entries()
cmudict_dict = {entry[0].lower(): entry[1] for entry in entries}
for entry in entries:
    phonemes = entry[1]
    all_phonemes.update(phonemes)
all_phonemes = list(all_phonemes)
all_phonemes.append('')
all_phonemes = fit_one_hot(all_phonemes)
phone_vector_dimension = len(all_phonemes.keys())


def get_filepaths(directory, format='.wav'):
    """
    load all file in the directory
    Parameters
    ----input-----
    directory: str. path of directory
    ----Return----
    file_paths: list. paths of files in the directory
    """
    file_paths = []  
    for root, _, files in os.walk(directory):
        for filename in files:
                filepath = os.path.join(root, filename)
                if filename.endswith(format):
                    file_paths.append(filepath)  
    return file_paths 

def remove_pun_except_apostrophe(input_string):
    """
    remove punctuations (except for ' ) of the inupt string.
    """    
    translator = str.maketrans('', '', string.punctuation.replace("'", ""))
    output_string = input_string.translate(translator).replace('  ',' ')
    return output_string

def remove_pun(input_string):
    """
    remove punctuations of the input_string.
    """    
    input_string = "".join([char for char in input_string if char not in string.punctuation])
    return input_string

def get_transcript(audio, whisper_model, return_seg=False):
    """
    get ASR result using whisper model

    Parameters
    ----input-----
    audio: Union[str, np.ndarray, torch.Tensor]
    whisper_model: 
        load whisper_model using:
        、、、
        import whisper
        whisper_model = whisper.load_model("base.en")
        、、、        
    return_seg: bool
        whether to return the segmentation result
    ----Return----
    transcript: str. ASR result of the input wavfile
    """
    result = whisper_model.transcribe(audio, fp16=False)
    if not return_seg:
        return result['text']
    else:
        return result

def convert_num_to_word(sen):
    """
    convert digit in a sentence to word. e.g. "7112" to "seven one one two".
    """
    try: #for 4 digit samples of speechocean data. 
        int(sen.replace(' ',''))
        sen = ' '.join([char for char in sen])
        sen = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in sen.split()])
        sen = sen.replace('  ',' ')
    except:
        sen = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in sen.split()])
    return sen


def get_phone_list(word_list):
    """
    convert word to phone using cmudict
    Parameters
    ----input-----
    word_list: list of word. e.g. [[word1],[word2]...] or [[word1, word2],...]
    ----Return----
    phone_list: list of corresponding phone e.g [[p1-1,p1-2], [p2-1,p2-2,p2-3],...] or [[p1-1,p1-2,p2-1,p2-2,p2-3], ...]
    """
    phone_list = []
    for word_position in word_list:
        the_phone_list = []
        for word in word_position:
            phone = cmudict_dict.get(word.lower(), '')
            the_phone_list.extend(phone)
        phone_list.append(the_phone_list)
        
    return phone_list

def get_phone_vector(phone_list):
    """
    convert phone to phone-vector using one-hot encoding
    Parameters
    ----input-----
    phone_list: list of phone. e.g. [[phone1-1, phone1-2],[phone2-1, phone2-2, phone2-3]...]
    ----Return----
    phone_vector: np.ndarray, [shape=(number of word, phone_vector_dimension=71)]
    """
    num_of_word = len(phone_list)
    phone_vector = np.zeros((num_of_word, phone_vector_dimension))

    for word_idx, the_phone_list in enumerate(phone_list):
        the_phone_vector = np.zeros((phone_vector_dimension, ))
        for phone in the_phone_list:
            the_phone_vector[all_phonemes[phone]]+=1
        phone_vector[word_idx,:] = the_phone_vector

    return phone_vector

def get_phone_features_from_wordlist(gt_word_list, asr_word_list):

    """
    get phone features of ground-turth word list and ASR word list
    Parameters
    ----input-----
    gt_word_list: list of ground-turth word. e.g. [[word1-gt],[word2-gt],[word3-gt]...]
    asr_word_list: list of asr word. e.g. [[word1-asr, word2-asr],[word3],[word4]...]
    # note: gt_word_list[i] is aligned with asr_word_list[i] based on the audio-word forced alignment result
    ----Return----
    phone_distance: np.ndarray, [shape=(number of ground-truth word, 1)]
        the phone distance (by SequenceMatcher) between ground-truth word and the asr-word
    phone_vector: np.ndarray, [shape=(number of ground-truth word, (phone_vector_dimension=71)*2)]
        the phone vector of ground-truth word and the asr-word
    phone_count: np.ndarray, [shape=(number of ground-truth word, 1)]
        the number of phones of ground-truth word divided by the number of phones of asr word
    """

    gt_length = len(gt_word_list)
    gt_phone_list = get_phone_list(gt_word_list)
    asr_phone_list = get_phone_list(asr_word_list)

    gt_phone_vector = get_phone_vector(gt_phone_list)
    asr_phone_vector = get_phone_vector(asr_phone_list)
    
    phone_distance = np.zeros((gt_length, 1))
    phone_count = np.zeros((gt_length, 1))
    for word_idx in range(gt_length):
        the_distance = SequenceMatcher(None, gt_phone_list[word_idx],asr_phone_list[word_idx])
        phone_distance[word_idx,0]=the_distance.ratio()
        if len(asr_phone_list[word_idx])!=0:
            phone_count[word_idx,0]=len(gt_phone_list[word_idx])/len(asr_phone_list[word_idx])

    phone_vector = np.concatenate((gt_phone_vector, asr_phone_vector), axis=1)

    return phone_distance, phone_vector, phone_count


def get_word_alignment_features(alignment_gt, alignment_asr):
    """
    get word-aligned features of ground-turth word list and ASR word list
    Parameters
    ----input-----
    alignment_gt: list, len = number of words in the ground-truth sentence
        word-audio alignment result of ground-truth sentence. e.g., [[(start_time1, end_time1, word1)], [(start_time2, end_time2, word2)], ...]
    alignment_asr: list, len = number of words in the ASR sentence
        word-audio alignment result of ASR sentence. e.g., [[(start_time1, end_time1, word1)], [(start_time2, end_time2, word2)], ...]

    ----Return----
    gt_word_list: list, len = number of words in the ground-truth sentence
        words in the ground-truth sentence, e.g. [[word1_gt],[word2_gt],[word3_gt],...]
    asr_word_list: list, len = number of words in the ground-truth sentence
        words in the ASR sentence aligned with ground-truth word. e.g. [[word1_asr, word2_asr],[word3_asr],...]
    alignment_features: np.ndarray, [shape=(number of words in the ground-truth sentence, 10)]
    phonevector: np.ndarray, [shape=(number of words in the ground-truth sentence, 71*2)]
        phonevector of the ground-truth words and asr words
    asr_wordidx_list: list, len = number of words in the ground-truth sentence
        mapping between ground-truth words and asr words. 
        asr_wordidx_list[i] = [j,m] means alignment_gt[i] is overlapped with alignment_asr[j] and alignment_asr[m]
    """
    gt_length = len(alignment_gt)
    
    gt_word_list = []
    asr_word_list = []
    asr_wordidx_list = []
    asr_distance = np.zeros((gt_length, 1))
    duration_gt = np.zeros((gt_length, 1))
    duration_asr = np.zeros((gt_length, 1))
    time_diff_start = np.zeros((gt_length, 1))
    time_diff_end = np.zeros((gt_length, 1))
    interval_gt = np.zeros((gt_length, 1))
    interval_asr = np.zeros((gt_length, 1))

    pre_end_gt = 0
    for gt_idx, gt_value in enumerate(alignment_gt):

        gt_word = gt_value[2]
        gt_start = float(gt_value[0])
        gt_end = float(gt_value[1])

        duration_gt[gt_idx,0] = (gt_end-gt_start)
        interval_gt[gt_idx,0] = (gt_start - pre_end_gt)
        pre_end_gt = gt_end

        gt_word_list.append([gt_word])

        asr_word_all = []
        asr_wordidx = []
        asr_start_flag = True
        the_asr_start = 0
        the_asr_end = float(alignment_asr[-1][1])

        pre_end_asr = 0
        asr_interval_list =  0
        for asr_idx, asr_value in enumerate(alignment_asr):
            asr_start =  float(asr_value[0])
            asr_end =  float(asr_value[1])
            if gt_end <= asr_start:
                break
            if gt_start >= asr_end:
                continue
            if max(gt_start, asr_start) <= min(gt_end, asr_end):
                asr_word =  asr_value[2]
                asr_word_all.append(asr_word)
                asr_wordidx.append(asr_idx)
                asr_interval_list += (asr_start - pre_end_asr)
                pre_end_asr = asr_end
                the_asr_end = asr_end
                if asr_start_flag:
                    the_asr_start = asr_start
                    asr_start_flag= False

        duration_asr[gt_idx,0] = (the_asr_end - the_asr_start)
        time_diff_start[gt_idx,0] = (the_asr_start - gt_start)
        time_diff_end[gt_idx,0] = (the_asr_end - gt_end)
        if len(asr_wordidx)!=0:
            interval_asr[gt_idx,0] = asr_interval_list/len(asr_wordidx)

        asr_wordidx_list.append(asr_wordidx)
        asr_word_list.append(asr_word_all)
        asr_distance[gt_idx,0] = ratio(gt_word, ' '.join(asr_word_all))*10

    align_word_count = [len(asr_word_list[word_idx]) for word_idx in range(gt_length)]
    phone_distance, phonevector, phone_count = get_phone_features_from_wordlist(gt_word_list, asr_word_list)

    align_word_count = np.asarray(align_word_count)
    align_word_count = np.expand_dims(align_word_count, axis=1)

    alignment_features = np.concatenate((asr_distance, align_word_count, duration_gt, duration_asr, time_diff_start, time_diff_end, phone_distance, phone_count, interval_gt, interval_asr), axis=1)
   
    return gt_word_list, asr_word_list, alignment_features, phonevector, asr_wordidx_list


def get_phone_alignment_features(pred_phones_gt, pred_phones_asr, pred_prob_gt, phone_ids_gt, pred_prob_asr, phone_ids_asr):
    """
    get phone-aligned features of ground-turth phone list and ASR phone list
    Parameters
    ----input-----
    pred_phones_gt: list, len = number of phones in the ground-truth sentence
        phoneme-audio alignment result of ground-truth sentence. e.g., [[(start_time1, end_time1, phone1)], [(start_time2, end_time2, phone2)], ...]
    pred_phones_asr: list, len = number of phones in the ASR sentence
        phoneme-audio alignment result of ASR sentence. e.g., [[(start_time1, end_time1, phone1)], [(start_time2, end_time2, phone2)], ...]
    pred_prob_gt: np.ndarray, [shape=(number of phones in the ground-truth sentence, 42)]
        output of the charsiu model
    phone_ids_gt: list, len = number of phones in the ground-truth sentence
        index of the aligned phone. 
    pred_prob_asr: np.ndarray, [shape=(number of phones in the ASR sentence, 42)]
        output of the charsiu model
    phone_ids_asr: list, len = number of phones in the ASR sentence
        index of the aligned phone. 

    ----Return----
    features: np.ndarray, [shape=(number of phones in the ground-truth sentence, 93)]
        extracted features. 
        features[:,:9] = alignment features
        features[:,9:9+42] = pred_prob_gt
        features[:,9+42:] = aligned_pred_prob_asr (align ASR phone to ground-truth phone using the time information)
    """
    gt_length = len(pred_phones_gt)
    
    asr_phone_list = []
    asr_phoneidx_list = []
    duration_gt = np.zeros((gt_length, 1))
    duration_asr = np.zeros((gt_length, 1))
    time_diff_start = np.zeros((gt_length, 1))
    time_diff_end = np.zeros((gt_length, 1))
    interval_gt = np.zeros((gt_length, 1))
    interval_asr = np.zeros((gt_length, 1))

    pre_end_gt = 0
    for gt_idx, gt_value in enumerate(pred_phones_gt):

        gt_start = gt_value[0]
        gt_end = gt_value[1]

        duration_gt[gt_idx,0] = (gt_end - gt_start)
        interval_gt[gt_idx,0] = (gt_start - pre_end_gt)
        pre_end_gt = gt_end

        asr_phone_all = []
        asr_phoneidx = []
        asr_start_flag = True
        the_asr_start = 0

        pre_end_asr = 0
        asr_interval_list =  0
        for asr_idx, asr_value in enumerate(pred_phones_asr):
            asr_start =  asr_value[0]
            asr_end =  asr_value[1]   
            if gt_end <= asr_start:
                break
            if gt_start >= asr_end:
                continue
            if max(gt_start, asr_start) <= min(gt_end, asr_end):
                asr_phone =  asr_value[2]
                asr_phone_all.append(asr_phone)
                asr_phoneidx.append(asr_idx)
                asr_interval_list += (asr_start - pre_end_asr)
                pre_end_asr = asr_end
                the_asr_end = asr_end
                if asr_start_flag:
                    the_asr_start = asr_start
                    asr_start_flag= False

        duration_asr[gt_idx,0] = (the_asr_end - the_asr_start)
        time_diff_start[gt_idx,0] = (the_asr_start - gt_start)
        time_diff_end[gt_idx,0] = (the_asr_end - gt_end)
        if len(asr_phoneidx)!=0:
            interval_asr[gt_idx,0] = asr_interval_list/len(asr_phoneidx)

        asr_phoneidx_list.append(asr_phoneidx)
        asr_phone_list.append(asr_phone_all)

    align_phone_count = [len(asr_phone_list[phone_idx]) for phone_idx in range(gt_length)]

    align_phone_count = np.asarray(align_phone_count)
    align_phone_count = np.expand_dims(align_phone_count, axis=1)

    the_gt_phone_prob = np.zeros((pred_prob_gt.shape[0], 1))
    the_asr_phone_prob = np.zeros((pred_prob_asr.shape[0], 1))

    the_gt_phone_prob[:, 0] = pred_prob_gt[np.arange(pred_prob_gt.shape[0]), phone_ids_gt]
    the_asr_phone_prob[:, 0] = pred_prob_asr[np.arange(pred_prob_asr.shape[0]), phone_ids_asr]

    aligned_asr_prob = np.zeros((pred_prob_gt.shape[0], 1))
    aligned_pred_prob_asr = np.zeros((pred_prob_gt.shape))
    for gt_p_idx, gt_p in enumerate(asr_phoneidx_list):
        aligned_asr_prob[gt_p_idx,:] = np.mean(the_asr_phone_prob[gt_p,:], axis=0)
        aligned_pred_prob_asr[gt_p_idx,:] = np.mean(pred_prob_asr[gt_p,:], axis=0)

    features = np.concatenate((align_phone_count, duration_gt, duration_asr, time_diff_start, time_diff_end, interval_gt, interval_asr, the_gt_phone_prob, aligned_asr_prob, pred_prob_gt, aligned_pred_prob_asr), axis=1)

    return features


def get_roberta_word_embed(word_list, num_of_token, roberta):
    """
    get roberta word embedding of input word list
    Parameters
    ----input-----
    word_list: list, len = number of words
    num_of_token: int
        number of the words in the sentence. 
    roberta: object.
        load roberta model using:
        、、、 
        from fairseq.models.roberta import RobertaModel
        roberta = RobertaModel.from_pretrained('./fairseq_roberta', checkpoint_file='model.pt')
        roberta.eval()
        、、、
    ----Return----
    sen_vec: np.ndarray, [shape=(num_of_token, 768)]
        word-by-word roberta embedding
    """
    sen_vec = np.zeros((num_of_token, 768))

    for w_idx, the_word_list in enumerate(word_list):
        the_sen = ' '.join(the_word_list)
        if the_sen=='':
            continue
        doc = roberta.extract_features_aligned_to_words(the_sen)
        the_sen_vec = np.zeros((768,))
        for tok in doc:
            if str(tok)=='<s>' or str(tok)=='</s>':
                continue
            the_vec = tok.vector.detach().numpy()
            the_sen_vec[:] += the_vec
        the_sen_vec /= len(the_word_list)
        sen_vec[w_idx,:] = the_sen_vec

    return sen_vec 

def get_charsiu_alignment(audio, sen, aligner):
    pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(audio=audio, text=sen)
    return pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map

def get_match_index(pred_words, words):    
    """
    remove [SIL] in the charsiu word alignment result.
    """
    selected_idx = []
    curren_idx=0
    for the_word in words:
        for i in range(curren_idx, len(pred_words)):
            if pred_words[i][2]==the_word:
                selected_idx.append(i)
                curren_idx = i+1
                break
    return selected_idx


def feature_extraction(audio, gt_sen, asr_sen, alignment_model, word_model):
    """
    extract features for the assessment model
    Parameters
    ----input-----
    audio: str. or np.ndarray [shape=(n,)]
        path to the input wavfile or np.ndarray of wavfile
    gt_sen: str.
        ground-truth sentence
    asr_sen: str.
        ASR sentence:
    alignment_model: object
        load charsiu model using:
        、、、 
        from Charsiu import charsiu_forced_aligner
            alignment_model = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
        、、、
    word_model: object:
        load roberta model using:
        、、、 
        from fairseq.models.roberta import RobertaModel
        word_model = RobertaModel.from_pretrained('./fairseq_roberta', checkpoint_file='model.pt')
        word_model.eval()
        、、、
    ----Return----
    """

    pred_phones_gt, pred_words_gt, words_gt, pred_prob_gt, phone_ids_gt, word_phone_map_gt = get_charsiu_alignment(audio, gt_sen, alignment_model)
    pred_phones_asr, pred_words_asr, words_asr, pred_prob_asr, phone_ids_asr, _ = get_charsiu_alignment(audio, asr_sen, alignment_model)

    features_p = get_phone_alignment_features(pred_phones_gt, pred_phones_asr, pred_prob_gt, phone_ids_gt, pred_prob_asr, phone_ids_asr)
    
    selected_idx_gt = get_match_index(pred_words_gt, words_gt)

    word_phone_map_gt = [word_phone_map_gt[i] for i in selected_idx_gt]

    pred_words_gt = np.asarray(pred_words_gt)
    pred_words_gt = pred_words_gt[selected_idx_gt]

    selected_idx_asr = get_match_index(pred_words_asr, words_asr)
    pred_words_asr = np.asarray(pred_words_asr)
    pred_words_asr = pred_words_asr[selected_idx_asr]
    gt_word_list, asr_word_list, features_w, phonevector, _ = get_word_alignment_features(pred_words_gt, pred_words_asr)
    
    num_of_token = len(selected_idx_gt)
    gt_word_embed =  get_roberta_word_embed(gt_word_list, num_of_token, word_model)
    asr_word_embed = get_roberta_word_embed(asr_word_list, num_of_token, word_model)
    
    return pred_words_gt, features_p, features_w, phonevector, gt_word_embed, asr_word_embed, word_phone_map_gt
