import os
import random
import matplotlib.pyplot as plt
import gc
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import RAdam
from transformers import BertConfig

from arguments import Arguments
from dataset import HealthFeatureDataset
from modules import BertForD2V

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        print('device: {}'.format(self.device))
        self.model_id = args.model_id
        
        self.collect_gb()
        self.fix_seed(123)
        
        self.train_dataloader = self.load_dataloader(args, 'train')
        self.val_dataloader = self.load_dataloader(args, 'val')
        self.config = self.load_model_config()
        self.model = self.load_model().to(self.device)
        self.optimizer = RAdam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99))
    
        now = datetime.now()
        self.date = now.strftime('%y%m%d-%H%M%S')
        print(self.date)
        self.res_dir = os.path.join(args.res_root_dir, 'pretrain', self.date)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        self.waiting = 0

    def run(self):
        tr_losses = []
        val_losses = []
        for epoch in range(self.args.epochs):
            tr_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            
            if epoch!=0 and epoch%20==0:    
                self.plot_loss_curve(epoch+1, tr_losses, val_losses)
                self.model.save_pretrained_model(self.res_dir, "{}_{}".format(self.date, epoch))
            self.waiting += 1

            if val_losses[-1] <= min(val_losses):
                self.waiting = 0
                self.model.save_pretrained_model(self.res_dir, "{}_{}".format(self.date, epoch), True)
                print("[!] The best checkpoint")
                
            b = True if self.waiting==0 else False
            self.write_log(self.args, 
                           best=b, 
                           epoch=epoch, 
                           tr_loss=tr_loss, 
                           val_loss=val_loss, 
                           val_losses=val_losses
                           )

            if self.waiting > self.args.patience:
                break        
        
        self.plot_loss_curve(epoch+1, tr_losses, val_losses)
            
    def train_epoch(self, epoch):
        self.model.train()
        tr_loss = 0
        for i, batch in enumerate(tqdm(self.train_dataloader, desc="[Epoch {}] TRAIN: ".format(epoch), mininterval=0.01)):
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            tr_loss += loss.item()

        return tr_loss/len(self.train_dataloader)
    
    def val_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader, desc="[Epoch {}] EVAL: ".format(epoch), mininterval=0.01)):
                outputs = self.model(batch)
                loss = outputs.loss
                self.optimizer.step()
                val_loss += loss.item()

        return val_loss/len(self.val_dataloader)
    
    def load_model_config(self):
        config = BertConfig.from_pretrained(self.args.model_id)
        config.pad_token_id = self.args.pad_token_id
        config.vocab_size = self.args.special_emb_size
        config.type_vocab_size = self.args.type_emb_size
        config.max_position_embeddings = self.args.max_len
        config.out_attentions = True
        config.output_hidden_states = True
        return config
    
    def load_model(self, ckpt_dir=None):
        model = BertForD2V(self.args, self.config, self.device)
        if ckpt_dir:
            ckpt = torch.load(ckpt_dir)
            model.load_state_dict(ckpt['model_state_dict'])
        return model
    
    def load_dataloader(self, args, split):
        if split == 'train':
            shuffle = True
            dir = args.train_data_dir
        else:
            shuffle = False
            if split == 'val':
                dir = args.val_data_dir                
            else:
                dir = args.test_data_dir
                
        dataset = HealthFeatureDataset(args, dir, split=split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
        return dataloader
    
    def save_model(self, losses): 
        self.model.save_pretrained(save_directory=os.path.join(self.res_dir, "{}.ckpt".format(self.date)))
        self.waiting += 1
        if losses[-1] <= min(losses):
            self.waiting = 0
            self.model.save_pretrained(save_directory=os.path.join(self.res_dir, "BEST_{}.ckpt".format(self.date)))
            print("[!] The best checkpoint")
            
    def write_log(self, args, best=False, **kwargs):
        fname = 'BEST_ckpt_log.txt' if best==True else 'ckpt_log.txt'
        with open(os.path.join(self.res_dir, fname), 'w') as log:
            log.write('Saved Epoch: {}\n'.format(kwargs['epoch']))
            log.write('Training Loss:   {}\n'.format(kwargs['tr_loss']))
            log.write('Validation Loss: {}\n'.format(kwargs['val_loss']))
            log.write('\n')
            log.write('Batch Size:  {}\n'.format(args.batch_size))
            log.write('Learning Rate:   {}\n'.format(args.lr))
            log.write('L1 loss Beta:    {}\n'.format(args.l1_beta))
            log.write('Base Model:  {}\n'.format(args.model_id))
            log.write('Hidden size: {}\n'.format(self.config.hidden_size))
            log.write('\n')
            log.write('Validation Loss list:\n')
            log.write(str(kwargs['val_losses']))
            
    def plot_loss_curve(self, epoch, tr_losses, val_losses):
        plt.plot(list(range(epoch)), tr_losses, label='Train Loss')
        plt.plot(list(range(epoch)), val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(list(range(epoch)))
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.res_dir, 'loss_{}.png'.format(epoch-1)))
        print("[!] The loss curve plotted")
        
    def collect_gb(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def fix_seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True  # for the final push
        torch.backends.cudnn.benchmark = False
        #np.random.set_state(st)  # st = np.randeom.get_state()
    

    