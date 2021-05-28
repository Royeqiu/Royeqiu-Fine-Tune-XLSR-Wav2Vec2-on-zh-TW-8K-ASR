import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from sklearn.model_selection import train_test_split
inference_label_path = './data/inference_label/inference_label.csv'
df = pd.read_csv(inference_label_path)
print(df.shape)
df=df.dropna()
train_df,test_df = train_test_split(df,train_size=0.8,random_state=1)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

class mt5_trainer():
    def __init__(self,args=None):

        self.model_args = T5Args()
        self.model_args.max_seq_length = 96
        self.model_args.train_batch_size = 20
        self.model_args.eval_batch_size = 20
        self.model_args.num_train_epochs = 2
        self.model_args.evaluate_during_training = True
        self.model_args.evaluate_during_training_steps = 30000
        self.model_args.use_multiprocessing = False
        self.model_args.fp16 = False
        self.model_args.save_steps = -1
        self.model_args.save_eval_checkpoints = False
        self.model_args.no_cache = True
        self.model_args.reprocess_input_data = True
        self.model_args.overwrite_output_dir = True
        self.model_args.preprocess_inputs = False
        self.model_args.num_return_sequences = 1
        self.model_args.output_dir = './'
        self.model = T5Model("mt5", "google/mt5-small", args=self.model_args)
    def train(self,train_df,test_df):
        # Train the model
        self.model.train_model(train_df, eval_data=test_df)

mt5_trainer = mt5_trainer()
mt5_trainer.train(train_df,test_df)

