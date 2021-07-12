import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import Series
import re

def chunkstring(x, length):
    chunks = len(x)
    l = [x[i:i+length] for i in range(0, chunks, length) ]
    return l

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer, library, indices):
        self.dataset = pd.read_csv(csv_path)
        self.library = library
        cols_n = self.dataset.columns.tolist()
        cols_n.reverse()
        types = list(self.dataset.type.unique())
        filter_links_phrases = ["https://", ".com", "http://", "youtube", "www"]
        self.dataset = pd.DataFrame(pd.concat([Series(row['type'], row['posts'].split("|||")) for _, row in self.dataset.iterrows()]).reset_index())
        [self.dataset.rename(columns = {name:cols_n[i]}, inplace = True) for i,name in enumerate(self.dataset.columns.tolist())]
        #self.dataset = self.dataset[~self.dataset['posts'].str.contains("|".join(filter_links_phrases))]
        #self.dataset.posts = self.dataset["posts"].str.lower()
        print(f"filter success {len(self.dataset)}")
        self.encodings = tokenizer(list(self.dataset['posts'].values), padding='max_length', truncation=True, max_length=32)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)
        


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        if self.library == "timm":
            AA = item["input_ids"]
            AA = AA.view(AA.size(0), -1).float()
            AA -= AA.min(1, keepdim=True)[0].clamp(1e-2)
            AA /= AA.max(1, keepdim=True)[0].clamp(1e-2)
            AA = torch.stack([AA for i in range(96)], dim=1)
            item["input_ids"] = AA.view(3, 128, 128)
        return item
    
    def __len__(self):
        return len(self._labels)

