from dataclasses import dataclass, field, fields, InitVar
from typing import Optional, Literal, List, Dict
# from validated_dc import ValidatedDC, get_errors, is_valid
import os
import torch


@dataclass
class Arguments():
    root_dir: Optional[str] = '.'
    data_root_dir: Optional[str] = os.path.join(root_dir, 'data')
    col_type_data_dir: Optional[str] = os.path.join(data_root_dir, 'column_types.xlsx')
    train_data_dir: Optional[str] = os.path.join(data_root_dir, 'pre_train.csv')
    val_data_dir: Optional[str] = os.path.join(data_root_dir, 'pre_val.csv')
    test_data_dir: Optional[str] = os.path.join(data_root_dir, 'pre_val.csv')
    res_root_dir: Optional[str] = os.path.join(root_dir, 'results')
    
    device: Optional[str] ='cuda' if torch.cuda.is_available() else 'cpu'
    model_id: Literal[
        'google/bert_uncased_L-2_H-128_A-2', 
        'google/bert_uncased_L-4_H-256_A-4',
        'bert-base-uncased'] = 'bert-base-uncased'
    
    pad_token_id: Optional[int] = -999
    mask_token_id: Optional[int] = -998
    
    special_emb_size: Optional[int] = 2  # 'mask', 'pad'
    type_emb_size: Optional[int] = 7
    max_len: Optional[int] = 177
    mask_ratio: Optional[float] = 0.15
    
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 4
    lr: Optional[float] = 1e-4
    patience: Optional[int] = 5

    l1_beta: Optional[float] = 4.0
    decay: Optional[float] = 0.999
    end_decay: Optional[float] = 0.9999
    anneal_end_step: Optional[int] = 300000
    skip_ls: List[str] = field(
        default_factory=list
    )

    metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.metrics = ['acc', 'f1', 'auc']

    def get_args_list(self):
        return [f.name for f in fields(args)]