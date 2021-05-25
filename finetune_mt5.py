from transformers import MT5ForConditionalGeneration, T5Tokenizer
import numpy as np
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
article = '''summarize: studies have shown that owning a dog is good for you'''
inputs = tokenizer(article,return_tensors='pt').input_ids
print(inputs)
outputs = model.generate(input_ids=inputs)
print(outputs)
print(tokenizer.decode(outputs[0]))