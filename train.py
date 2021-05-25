from src.DataCollatorCTCWithPadding import DataCollatorCTCWithPadding
from src.Audio_Processor import Audio_Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from datasets import load_metric,load_from_disk
import numpy as np
from transformers import Trainer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.set_device(torch.device('cuda:0'))
processor_save_path = './processor'
datasets_path = './training_data'
audio_processor = Audio_Processor(processor_save_path=processor_save_path)
data_collator = DataCollatorCTCWithPadding(processor=audio_processor.processor, padding=True)
wer_metric = load_metric("wer")

def load_datasets(datasets_path):
    train_datasets = load_from_disk(os.path.join(datasets_path,'train_datasets.ds'))
    test_datasets = load_from_disk(os.path.join(datasets_path, 'test_datasets.ds'))
    return train_datasets,test_datasets

def compute_metrics(pred):

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = audio_processor.processor.tokenizer.pad_token_id

    pred_str = audio_processor.processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = audio_processor.processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def remove_long_common_voicedata(dataset, max_seconds=6):
    dftest = dataset.to_pandas()
    dftest['len'] = dftest['input_values'].apply(len)
    maxLength = max_seconds * 16000
    dftest = dftest[dftest['len'] < maxLength]
    dftest = dftest.drop('len', 1)
    dataset = dataset.from_pandas(dftest)
    del dftest
    return dataset


if __name__ == '__main__':
    train_datasets, test_datasets = load_datasets(datasets_path=datasets_path)
    train_datasets = remove_long_common_voicedata(train_datasets)
    test_datasets = remove_long_common_voicedata(test_datasets)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=audio_processor.processor.tokenizer.pad_token_id,
        vocab_size=len(audio_processor.processor.tokenizer)
    )

    training_args = TrainingArguments(
      output_dir="./train_model/wav2vec2-large-xlsr-zh_TW-8K-demo",
      group_by_length=False,
      per_device_train_batch_size=1,
      gradient_accumulation_steps=1,
      evaluation_strategy="steps",
      eval_accumulation_steps= 1,
      num_train_epochs=30,
      fp16=True,
      save_steps=400,
      eval_steps=400,
      logging_steps=400,
      learning_rate=3e-4,
      warmup_steps=500,
      save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_datasets,
        eval_dataset=test_datasets,
        tokenizer=audio_processor.feature_extractor,
    )
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()