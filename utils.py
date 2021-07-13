import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.dataset = pd.read_csv(csv_path)
        self.library = library
        cols_n = self.dataset.columns.tolist()
        cols_n.reverse()
        types = list(np.vectorize(lambda x: x.lower())(self.dataset["type"].unique()))
        filter_links_phrases = ["https://", ".com", "http://", "youtube", "www"]
        self.dataset = pd.DataFrame(pd.concat([Series(row['type'], row['posts'].split("|||")) for _, row in self.dataset.iterrows()]).reset_index())
        [self.dataset.rename(columns = {name:cols_n[i]}, inplace = True) for i,name in enumerate(self.dataset.columns.tolist())]
        self.dataset = self.dataset[self.dataset['posts'].str.split(" ").str.len() > 32]
        print("Data before filtering:"+str(len(self.dataset)))
        self.dataset.posts = self.dataset["posts"].str.lower()
        self.dataset = self.dataset[~self.dataset['posts'].str.contains(pat = "|".join(filter_links_phrases),regex=True)]
        self.dataset = self.dataset[~self.dataset['posts'].str.contains(pat = "|".join(types), regex=True)]
        print(f"filter success {len(self.dataset)}")
        self.encodings = tokenizer(list(self.dataset['posts'].values), padding='max_length', truncation=True, max_length=32)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        return item
    
    def __len__(self):
        return len(self._labels)

