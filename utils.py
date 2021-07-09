import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import Series
import re

def chunkstring(string, length):
  return re.findall('.{%d}' % length, string)

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer, library, long):
        
        self.dataset = pd.read_csv(csv_path)
        self.long = long
        self.library = library
        self.dataset = pd.concat([Series(self.dataset['type'], chunkstring(row['posts'], 512)) for _, row in self.dataset.iterrows()]).reset_index()
        self.encodings = tokenizer(list(self.dataset['posts'].values), max_length=64*64 if library == "timm" else 512, truncation=True, padding='max_length', return_attention_mask=True)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        if self.library == "timm":
            AA = item["input_ids"]
            AA = AA.view(AA.size(0), -1).float()
            AA = torch.stack([AA for i in range(3)], dim=1)
            AA -= AA.min(1, keepdim=True)[0].clamp(1e-2)
            AA /= AA.max(1, keepdim=True)[0].clamp(1e-2)
            item["input_ids"] = AA.view(3, 64, 64)
        return item
    
    def __len__(self):
        return len(self._labels)

