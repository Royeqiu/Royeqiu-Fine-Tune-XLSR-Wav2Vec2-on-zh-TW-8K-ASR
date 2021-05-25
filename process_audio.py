# coding=UTF-8

from datasets import load_dataset
import os
from src.Hypothesis_Processor import Hypothesis_Processor
from src.Audio_Processor import Audio_Processor
lng_code = 'zh-TW'
dataset_name = 'common_voice'
vocab_save_path = './processor/vocab.json'
audio_save_path = './training_data'
processor_save_dir = './processor'
def process_audio(common_voice_train,common_voice_test,audio_save_path,processor_save_dir = None):

    audio_processor = Audio_Processor(vocab_path = vocab_save_path)
    if processor_save_dir is not None:
        audio_processor.save_processor(processor_save_dir = processor_save_dir)
    common_voice_audio_train = audio_processor.process_audio(common_voice_train,os.path.join(audio_save_path,'train_datasets.ds'))
    common_voice_audio_test = audio_processor.process_audio(common_voice_test,os.path.join(audio_save_path,'test_datasets.ds'))
    return common_voice_audio_train,common_voice_audio_test



def process_hypothesis(common_voice_train,common_voice_test,vocab_save_path=vocab_save_path):
    hypo_processor = Hypothesis_Processor()
    common_voice_train,vocab_train = hypo_processor.extract_vocab(common_voice_train)
    common_voice_test,vocab_test = hypo_processor.extract_vocab(common_voice_test)
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = hypo_processor.build_vocab(vocab_list)
    hypo_processor.save_vocab(vocab_dict,vocab_save_path)
    return common_voice_train,common_voice_test
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    common_voice_train = load_dataset(dataset_name, lng_code, split="train+validation")
    common_voice_test = load_dataset(dataset_name, lng_code, split="test")
    common_voice_train,common_voice_test = process_hypothesis(common_voice_train,common_voice_test,vocab_save_path)
    process_audio(common_voice_train, common_voice_test, audio_save_path, processor_save_dir)
