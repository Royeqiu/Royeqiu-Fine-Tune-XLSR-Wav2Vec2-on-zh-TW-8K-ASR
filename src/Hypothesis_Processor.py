import re
import json


class Hypothesis_Processor():

    def __init__(self, chars_to_ignore_regex='[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'):
        self.chars_to_ignore_regex = chars_to_ignore_regex

    def extract_vocab(self, datasets):
        def remove_special_characters(batch):
            batch["sentence"] = re.sub(self.chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
            return batch

        def extract_all_chars(batch):
            all_text = " ".join(batch["sentence"])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}

        datasets = datasets.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        datasets = datasets.map(remove_special_characters)
        vocab = datasets.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                             remove_columns=datasets.column_names)
        return datasets, vocab

    def build_vocab(self, vocab_list):
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        return vocab_dict

    def save_vocab(self, vocab_dict, filename='./processor/vocab.json'):
        with open(filename, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
