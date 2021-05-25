# coding=UTF-8
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import torchaudio
import os
import librosa
import numpy as np
from datasets.arrow_dataset import Dataset

class Audio_Processor():
    def __init__(self, vocab_path="./vocab.json", original_sample_rate = 48000, target_sample_rate = 16000, processor_save_path = None):

        if processor_save_path is None:
            self.tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]",
                                                  word_delimiter_token="|")
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                              do_normalize=True, return_attention_mask=True)
            self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

        else:
            self.load_processor(processor_save_path)

        self.original_sample_rate = original_sample_rate
        self.target_sample_rate = target_sample_rate
    def process_audio(self, datasets,path = None):
        def speech_file_to_array_fn(batch):
            speech_array, sampling_rate = torchaudio.load(batch["path"])
            batch["speech"] = speech_array[0].numpy()
            batch["sampling_rate"] = sampling_rate
            batch["target_text"] = batch["sentence"]
            return batch

        def resample(batch):
            batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
            batch["sampling_rate"] = 16_000
            return batch
        def prepare_dataset(batch):
            # check that all files have the correct sampling rate
            assert (
                    len(set(batch["sampling_rate"])) == 1
            ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

            batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["target_text"]).input_ids
            return batch
        def save_audio_datasets(datasets,path):
            datasets.save_to_disk(path)
        datasets = Dataset.from_dict(datasets[12:14]).map(speech_file_to_array_fn, remove_columns=datasets.column_names)
        datasets = datasets.map(resample, num_proc=4)
        datasets = datasets.map(prepare_dataset, remove_columns = datasets.column_names, batch_size=8, num_proc=4, batched=True)
        if path is not None:
            save_audio_datasets(datasets,path)

        return datasets

    def save_processor(self,processor_save_dir):
        self.processor.save_pretrained(processor_save_dir)

    def load_processor(self,processor_save_dir):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_save_dir)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor