import torch
from torch.utils.data import Dataset
import pandas as pd

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer, library):
        self.dataset = pd.read_csv(csv_path)
        self.library = library
        self.encodings = tokenizer(list(self.dataset['posts'].values), max_length=32*32*3 if library == "timm" else 512, truncation=True, padding="longest", return_attention_mask=True)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        if self.library == "timm":
            AA = item["input_ids"].view(item["input_ids"].size(0), -1).float()
            AA -= AA.min(1, keepdim=True)[0].clamp(1e-2)
            AA /= AA.max(1, keepdim=True)[0].clamp(1e-2)
            item["input_ids"] = AA.view(3, 32, 32)
        return item
    
    def __len__(self):
        return len(self._labels)

