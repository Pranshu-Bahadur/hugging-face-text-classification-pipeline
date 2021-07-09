import torch
from torch.utils.data import Dataset
import pandas as pd

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer, library):
        self.dataset = pd.read_csv(csv_path)
        self.library = library
        self.encodings = tokenizer(list(self.dataset['posts'].values), max_length=768, truncation=True, padding="longest", return_attention_mask=True)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        if self.library == "timm":
            item["input_ids"] -= item["input_ids"].min(1, keepdim=True)[0]
            item["input_ids"] /= item["input_ids"].min(1, keepdim=True)[0]
            item["input_ids"] = item["input_ids"].view(3, 256).float()
            item["attention_mask"] = item["attention_mask"].view(3, 256).float()
        return item
    
    def __len__(self):
        return len(self._labels)

