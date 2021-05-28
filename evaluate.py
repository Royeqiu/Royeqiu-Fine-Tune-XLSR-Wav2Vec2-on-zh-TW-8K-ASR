import torch
import torchaudio
from datasets import load_dataset, load_metric,load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
#becasue Windows can't process mp3 by using librosa, the data is loaded from local. The data is processed by other computer.
'''
test_dataset = load_dataset("common_voice", 'zh-TW', split="test") #TODO: replace {lang_id} in your language code here. Make sure the code is one of the *ISO codes* of [this](https://huggingface.co/languages) site.
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\：\；\。\，]'  # TODO: adapt this list to include all special characters you removed from the data
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)
'''
test_dataset=load_from_disk('training_data/test_datasets.ds')
wer = load_metric("wer")
processor = Wav2Vec2Processor.from_pretrained('./processor') #TODO: replace {model_id} with your model id. The model id consists of {your_username}/{your_modelname}, *e.g.* `elgeish/wav2vec2-large-xlsr-53-arabic`
model = Wav2Vec2ForCTC.from_pretrained('./train_model/wav2vec2-large-xlsr-zh_TW-8K-demo') #TODO: replace {model_id} with your model id. The model id consists of {your_username}/{your_modelname}, *e.g.* `elgeish/wav2vec2-large-xlsr-53-arabic`
model.to("cuda")


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch
#print(test_dataset["labels"])
#print(processor.batch_decode(test_dataset["labels"]))
result = test_dataset.map(evaluate, batched=True, batch_size=8)
print(result["pred_strings"][0],processor.decode(result["labels"][0]))
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=processor.batch_decode(result["labels"]))))