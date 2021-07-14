import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import Series
import re
import numpy as np

def chunkstring(x, length):
    return re.findall('.{%d}'%length, x)

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.dataset = pd.read_csv(csv_path)
        cols_n = self.dataset.columns.tolist()
        cols_n.reverse()
        types = list(np.vectorize(lambda x: x.lower())(self.dataset["type"].unique()))
        self.dataset['posts'] = self.dataset['posts'].str.lower()
        self.dataset['posts'] = self.dataset['posts'].str.replace(r'[|||]', '')
        self.dataset['posts'] = self.dataset['posts'].str.replace(r'|\b'.join(types), '')
        self.dataset['posts'] = self.dataset['posts'].str.replace(r'\bhttp.*[a-zA-Z0-9]\b', '')
        self.dataset = self.dataset[self.dataset['posts'].map(len)>32]
        #print("Exploding posts and types...\n")
        #self.dataset = self.dataset[self.dataset['total_words']>256]
        #self.dataset = pd.DataFrame(pd.concat([Series(row['type'],row['posts'].split('|||')) for _, row in self.dataset.iterrows()]).reset_index())
        #self.dataset = self.dataset.rename(columns={k: cols_n[i] for i,k in enumerate(list(self.dataset.columns))})
        self.dataset = pd.DataFrame(pd.concat([Series(row['type'], chunkstring(row['posts'], 128)) for _, row in self.dataset.iterrows()]).reset_index())
        self.dataset = self.dataset.rename(columns={k: cols_n[i] for i,k in enumerate(list(self.dataset.columns))})
        self.dataset['total'] = self.dataset['posts'].str.split()
        self.dataset['total'] = self.dataset['total'].map(len)
        #self.dataset = self.dataset[self.dataset['total']>=30]
        #self.dataset = self.dataset[self.dataset['total']<=40]
        print(self.dataset.head())
        print(f"filter success {len(self.dataset)}")
        print("Mean, mode, max, min lengths:\n")
        print(self.dataset['total'].mean())
        print(self.dataset.total.value_counts())
        print(max(self.dataset['total']))
        print(min(self.dataset['total']))
        self.dataset.drop(columns=['total'])

        self.distribution = self.dataset.type.value_counts()
        print(f'Dataset imbalanced distribution :\n{dict(self.distribution)}')
        max_size = self.distribution.max()        
        #https://stackoverflow.com/questions/48373088/duplicating-training-examples-to-handle-class-imbalance-in-a-pandas-data-frame
        
        lst = [self.dataset]
        for class_index, group in self.dataset.groupby('type'):
            lst.append(group.sample(max_size - len(group), replace=True))
        self.dataset = pd.concat(lst)
        self.distribution = dict(self.dataset.type.value_counts())
        print(f'Dataset balanced distribution after oversampling:\n{self.distribution}')
        print(f"Total samples after balancing:\n\n {len(self.dataset)}\n\n\n")
        
        print(f"Tokenizing dataset...")
        #TODO add a Debug mode.
        self.encodings = tokenizer(list(self.dataset['posts'].values), padding=True, truncation=True)
        print(f"Tokenizing complete.\n\n")
        keep_these_keys = ["input_ids", "attention_mask"]
        self.encodings = {k:v for k,v in list(self.encodings.items()) if k in keep_these_keys}
        self.labels = {k: v for v, k in enumerate(self.distribution.keys())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        x = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        y = torch.tensor(self._labels[idx])
        x["labels"] = y
        return x
    
    def __len__(self):
        return len(self._labels)

