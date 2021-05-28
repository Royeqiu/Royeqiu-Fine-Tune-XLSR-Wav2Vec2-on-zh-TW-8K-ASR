from datasets import load_dataset, load_metric,load_from_disk
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model, T5Args
model_dir = 'train_model/mt5-2'
inference_label_path = './data/inference_label/inference_label.csv'
df = pd.read_csv(inference_label_path)
df=df.dropna()
train_df,test_df = train_test_split(df,train_size=0.8,random_state=1)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
model_args = T5Args()
model_args.max_length = 48
model_args.length_penalty = 1
model_args.num_beams = 10
model = T5Model("mt5", model_dir, args=model_args,use_cuda=False)
wer = load_metric("wer")
res=model.predict(test_df.apply(lambda x:x['prefix']+ ': ' + x['input_text'],  axis=1).to_list())
wer_score = wer.compute(predictions=res, references=test_df['target_text'])
print('wer score: {} and word correct rate: {}'.format(wer_score,(1-wer_score)*100))