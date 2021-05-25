#encoding = utf-8
import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelWithLMHead
)
import pandas as pd
import torch
import os

model_name = "voidful/wav2vec2-large-xlsr-53-tw-gpt"
device = "cuda"
processor_name = "voidful/wav2vec2-large-xlsr-53-tw-gpt"
chars_to_ignore_regex = '''[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]'''
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(processor_name)

tokenizer = AutoTokenizer.from_pretrained("ckiplab/gpt2-base-chinese")
gpt_model = AutoModelWithLMHead.from_pretrained("ckiplab/gpt2-base-chinese").to(device)

resampler = torchaudio.transforms.Resample(orig_freq=16_000, new_freq=16_000)

def load_file_to_data(file):
    batch = {}
    speech, _ = torchaudio.load(file)
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    return batch

def predict(data):
    features = processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    decoded_results = []
    for logit in logits:
        pred_ids = torch.argmax(logit, dim=-1)
        mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
        vocab_size = logit.size()[-1]
        voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1,vocab_size)),dim=-1)
        gpt_input = torch.cat((torch.tensor([tokenizer.cls_token_id]).to(device),pred_ids[pred_ids>0]), 0)
        gpt_prob = torch.nn.functional.softmax(gpt_model(gpt_input).logits, dim=-1)[:voice_prob.size()[0],:]
        comb_pred_ids = torch.argmax(gpt_prob*voice_prob, dim=-1)
        decoded_results.append(processor.decode(comb_pred_ids))

    return decoded_results
path = 'data/converted_data'
files = os.listdir(path)
predicted_text = []

for index,file in enumerate(files):
    if index % 100 ==0:
        print(len(files),index)
    predicted_text.append(predict(load_file_to_data(os.path.join(path,file))))
pd.DataFrame({'file':files,'inference':predicted_text}).to_csv('./data/w2v_inference.csv',index=False)