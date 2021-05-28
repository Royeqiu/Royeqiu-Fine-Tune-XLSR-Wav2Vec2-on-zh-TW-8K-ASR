import pandas as pd
import json
import opencc
infereced_path = './data/w2v_inference.csv'
manifest_path = './data/manifest/train_common_tw_manifest.pkl'
train_manifest_path = './data/manifest/train_common_tw_manifest.pkl'
test_manifest_path = './data/manifest/test_common_tw_manifest.pkl'
inference_label_path = './data/inference_label/inference_label.csv'
def load_inferenced_data(infereced_path, is_convert_tranditional = False):
    if is_convert_tranditional:
        converter = opencc.OpenCC('t2s.json')
        def process_inference(x):
            obj = json.loads(x.replace('\'','"'))[0]
            return converter.convert(obj)
    else:
        def process_inference(x):
            obj = json.loads(x.replace('\'','"'))[0]
            return obj
    df = pd.read_csv(infereced_path)
    df['inference'] = df['inference'].apply(process_inference)

    return df

def load_manifest_data(manifest_path,is_convert_tranditional=False):
    fp = open(manifest_path,'r',encoding='utf-8')
    files = []
    durations = []
    texts = []
    if is_convert_tranditional:
        converter = opencc.OpenCC('s2tw.json')
    for data in fp:
        audio_meta = json.loads(data.strip())
        files.append(audio_meta['audio_filepath'].split('\\')[-1])
        durations.append(audio_meta['duration'])
        if is_convert_tranditional:
            texts.append(converter.convert(audio_meta['text']))
        else:
            texts.append(audio_meta['text'])
    return pd.DataFrame({'file':files,'duration':durations,'text':texts})
if __name__ == '__main__':
    train_manifest_df = load_manifest_data(train_manifest_path,is_convert_tranditional=True)
    test_manifest_df = load_manifest_data(test_manifest_path,is_convert_tranditional=True)
    total_manifest_df = pd.concat([train_manifest_df,test_manifest_df])
    inferenced_df = load_inferenced_data(infereced_path)
    print(inferenced_df.shape)
    print(inferenced_df.head())
    print(total_manifest_df.head())
    dataset_df = inferenced_df.merge(total_manifest_df,on=['file'])
    print(dataset_df.shape)
    print(dataset_df.head())
    dataset_df = dataset_df.rename(columns={'inference':'input_text','text':'target_text'})[['input_text','target_text']]
    dataset_df['prefix'] = ['error word correction'] * dataset_df.shape[0]
    print(dataset_df.shape)
    dataset_df = dataset_df.dropna().reset_index(drop = True)
    print(dataset_df.shape)
    dataset_df.to_csv(inference_label_path,index=False)

    print(dataset_df.head())