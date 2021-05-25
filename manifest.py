#process common_voice_data and use openCC convert simplified chinese to traditional chinese.
#The common voice data can be downloaded on https://commonvoice.mozilla.org/zh-TW/datasets.
#Windows doesn't support mp3 parsing so mp3 needs to be converted into wav by using ffmpeg.
import pandas as pd
import librosa
import os
import pickle
import logging
import json
import opencc

from sklearn.model_selection import train_test_split
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
transcript_path = 'data/transcript'
wav_path = 'data/converted_data'
transcript_id = 'transcript_id'
PATH = 'path'
AUDIO_FILEPATH = 'audio_filepath'
DURATION = 'duration'
FILE_NAME = 'file_name'
SENTENCE = 'sentence'
TEXT = 'text'
write_file_path = 'data/manifest/'
train_write_file = 'train_common_tw_manifest.pkl'
test_write_file = 'test_common_tw_manifest.pkl'
def load_transcript_df(path):
    files = [file for file in os.listdir(path) if '.tsv' in file ]
    total_df = []
    for file in files:
        df = pd.read_csv(os.path.join(path,file),delimiter='\t')
        total_df.append(df)
    transcript_df = pd.concat(total_df)
    transcript_df[transcript_id] = transcript_df[PATH].apply(lambda x:x.split('.')[0])
    logging.info('Loading transcript_df finished')
    return transcript_df

def load_wav_df(wav_path):

    files = [file for file in os.listdir(wav_path) if '.wav' in file]
    path_list = []
    duration_list = []
    for index,file in enumerate(files):
        if index % 200 == 0:
            logging.info(f'{index} wav file has been loaded')
        audio_filepath = os.path.join(wav_path, file)
        duration = librosa.core.get_duration(filename=audio_filepath)
        path_list.append(audio_filepath)
        duration_list.append(duration)
    wav_df = pd.DataFrame({FILE_NAME: files, AUDIO_FILEPATH: path_list, DURATION: duration_list})
    wav_df[transcript_id] = wav_df[FILE_NAME].apply(lambda x:x.split('.')[0])
    logging.info('Loading wav_df finished')
    return wav_df

def write_manifest(manifest_list,path):
    wrote_f = open(path,'w',encoding='utf-8')
    for manifest in manifest_list:
        wrote_f.write(json.dumps(manifest, ensure_ascii=False) + '\n')
    wrote_f.close()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    converter = opencc.OpenCC('t2s.json')
    transcript_df = load_transcript_df(transcript_path)
    wav_df = load_wav_df(wav_path)
    combined_df = pd.merge(right=transcript_df,left=wav_df, on=transcript_id)
    combined_df[TEXT] = combined_df[SENTENCE].apply(lambda x,converter:converter.convert(x),args=(converter,))

    manifest_info = []
    for audio_filepath, duration, text in zip(combined_df[AUDIO_FILEPATH], combined_df[DURATION], combined_df[TEXT]):
        manifest_info.append({AUDIO_FILEPATH:audio_filepath,DURATION:duration,TEXT:text})
    logging.info(f'There are {len(manifest_info)} audio files')
    train_manifest,test_manifest=train_test_split(manifest_info,test_size = 0.2,random_state = 1)
    logging.info(f'training size: {len(train_manifest)}')
    logging.info(f'testing size: {len(test_manifest)}')
    print(json.dumps(train_manifest[0]))
    write_manifest(train_manifest,os.path.join(write_file_path,train_write_file))
    write_manifest(test_manifest, os.path.join(write_file_path, test_write_file))


