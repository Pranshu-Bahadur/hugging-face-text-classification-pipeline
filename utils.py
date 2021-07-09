import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import Series
import re

def chunkstring(string, length):
  return re.findall('.{%d}' % length, string)

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

class SpreadSheetNLPCustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer, library, long):
        
        self.dataset = pd.read_csv(csv_path)
        self.long = long
        self.library = library
        cols_n = self.dataset.columns.tolist()
        print(self.dataset.head())
        self.dataset = pd.DataFrame(pd.concat([Series(row['type'], chunkstring(row['posts'], 512)) for _, row in self.dataset.iterrows()]).reset_index())
        print(self.dataset.head())
        [self.dataset.rename(columns = {name:cols_n[i]}, inplace = True) for i,name in enumerate(self.dataset.columns.tolist())]
        self.encodings = tokenizer(list(self.dataset['posts'].values), max_length=512, truncation=True, padding='max_length', return_attention_mask=True)
        self.labels = {k: v for v, k in enumerate(self.dataset.type.unique())}
        self.dataset['type'] = self.dataset['type'].apply(lambda x: self.labels[x])
        self._labels = list(self.dataset['type'].values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self._labels[idx])
        if self.library == "timm":
            AA = item["input_ids"]
            AA = AA.view(AA.size(0), -1).float()
            AA = torch.stack([AA for i in range(8*3)], dim=1)
            AA -= AA.min(1, keepdim=True)[0].clamp(1e-2)
            AA /= AA.max(1, keepdim=True)[0].clamp(1e-2)
            item["input_ids"] = AA.view(3, 64, 64)
        return item
    
    def __len__(self):
        return len(self._labels)

