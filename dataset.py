import pandas as pd
import numpy as np
import os
import copy


import torch
from torch.utils.data import Dataset

from arguments import Arguments

class HealthFeatureDataset(Dataset):
    def __init__(self, args, dir, split='train'):
        self.args = args
        self.data = pd.read_csv(dir, low_memory=False)
        self.type_ids = self.get_type_ids()
        self.max_len = args.max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        id = self.data.iloc[idx]['EID']
        sequence = self.data.iloc[idx, 1:]
        sequence = sequence.fillna(-999)
        
        masked_ids, masked_idx = self.get_masked_ids(list(sequence), len(sequence))
        sequence = torch.tensor(sequence, dtype=torch.float32).to(self.args.device)
        masked_sequence = torch.tensor(masked_ids, dtype=torch.float32).to(self.args.device)
        
        type_ids = torch.tensor(self.type_ids).to(self.args.device)
        attn_masks = torch.tensor([1]*len(sequence), dtype=torch.int64).to(self.args.device)
        attn_masks[sequence==float(-999)] = 0
        
        out_dict = {
            'id': id,
            'input_ids': sequence,
            'masked_ids': masked_sequence,
            'token_type_ids': type_ids,
            'attention_mask': attn_masks,
        }
        return out_dict
    
    def get_type_ids(self):
        df_type = pd.read_excel(self.args.col_type_data_dir)
        column_lst = list(self.data.columns)[1:]
        type_lst = list(df_type.columns)

        types, type_ids = [], []
        for col in column_lst:
            typ = df_type.columns[df_type.isin([col]).any()].tolist()[0]
            # types.append(typ)
            type_ids.append(type_lst.index(typ))
            
        return type_ids
    
    def get_masked_ids(self, input_ids, input_len):
        num_mask = int(input_len*self.args.mask_ratio)
        masked_ids = copy.deepcopy(input_ids)
        labels = torch.tensor([-100]*self.max_len)

        masked_idx = [i for i, t in enumerate(masked_ids) if t < 0]
        masked_idx = np.random.choice(masked_idx, num_mask, replace=False)
        
        for i in masked_idx:
            masked_ids[i] = self.args.mask_token_id
        return masked_ids, masked_idx

if __name__ == '__main__':
    args = Arguments()
    dir = './data/Demo_Ins0_final.csv'
    data = HealthFeatureDataset(args, dir)
    print(len(data))
    print(data[10])
