import os
import librosa
import soundfile as sf
from tqdm import tqdm

def get_filepaths(directory):
      file_paths = []  
      for root, directories, files in os.walk(directory):
            for filename in files:
                  # Join the two strings in order to form the full filepath.
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.WAV'):
                        file_paths.append(filepath)  
      return file_paths

inputdir = './WAVE'
outputdir = './wav'

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

file_list = get_filepaths(inputdir)


for path in tqdm(file_list):
      wavname = path.split(os.sep)[-1]
      new_path = os.path.join(outputdir, wavname.replace('.WAV','.wav'))
      if os.path.isfile(new_path):
            continue
      y, rate = librosa.load(path, sr=16000)
      sf.write(new_path, y, rate)
