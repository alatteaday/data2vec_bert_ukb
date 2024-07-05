import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel, BertModel,
    
)
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class HCEmbeddings(nn.Module):
    def __init__(self, in_dim, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.linear = nn.Linear(in_dim, config.hidden_size)
        self.special_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    
    def init_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)
        self.apply(weights_init)

    def forward(self, 
                input_ids, 
                token_type_ids, 
                position_ids=None, 
                past_key_values_length=0):
        
        input_ids_expanded = input_ids.unsqueeze(dim=2)
        embeddings = self.linear(input_ids_expanded)
        embeddings[torch.where(input_ids==-999)] = self.special_embeddings(torch.tensor(0).to(self.device))
        if input_ids[input_ids==-998].any():
            embeddings[torch.where(input_ids==-998)] = self.special_embeddings(torch.tensor(1).to(self.device))
        
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:input_ids.size()[1]+past_key_values_length]
       
        type_embeddings = self.type_embeddings(token_type_ids)
        embeddings += type_embeddings
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertModelHC(BertPreTrainedModel):
    def __init__(self, config, device, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = HCEmbeddings(1, config, device)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.post_init()
        
    def forward(self,
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        
        batch_size, seq_length = input_shape
        device = input_ids.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length,
        )
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length+past_key_values_length), device=device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            past_key_values=past_key_values,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        

class BertForD2V(nn.Module):
    def __init__(self, args, config, device, test=False):
        super(BertForD2V, self).__init__()
        self.args = args
        self.device = device

        self.model = BertModelHC(config, device)
        self.model2 = deepcopy(self.model)
        self.reg_head = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_updates = 0
        # self.decay = args.decay
        self.test = test
        
    def forward(self, batch):
        outputs = self.model(
            input_ids=batch['masked_ids'], 
            attention_mask=batch['attention_mask'], 
            token_type_ids=batch['token_type_ids']
        )
        pred = outputs.last_hidden_state
        masks = batch['masked_ids'] == self.args.mask_token_id
        
        if not self.test:
            self._update_params(self.args)
        
        self.model2.eval()
        with torch.no_grad():    
            outputs2 = self.model2(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                token_type_ids=batch['token_type_ids']
            )
            tgt = outputs2.hidden_states
            tgt = tgt[-self.model.config.num_hidden_layers:]
            tgt = sum(tgt) / len(tgt)
            tgt = self.layer_norm(tgt)
            
        pred = pred[masks]
        tgt = tgt[masks]
        pred = self.reg_head(pred)
        loss_fct = nn.SmoothL1Loss(reduction='none', beta=self.args.l1_beta)
        loss = loss_fct(tgt, pred)
        loss = loss.sum(dim=-1).sum().div(pred.size(0))
        
        out = MultimodalMRBertPretrainOutput(
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            pooler_output=outputs.pooler_output,
            tgt_last_hidden_state=outputs2.last_hidden_state,
            tgt_hidden_states=outputs2.hidden_states,
            tgt_pooler_output=outputs2.pooler_output
        )
        return out
    
    def _update_params(self, args):
        if args.decay != args.end_decay:
            if self.num_updates >= args.anneal_end_step:
                decay = args.end_decay
            else:
                decay = self._get_decay(args)
            self.decay = decay
        if self.decay < 1:
            state_dict = {}
            params2 = self.model2.state_dict()
            for key, param in self.model.state_dict().items():
                param2 = params2[key].float()
                if key in args.skip_ls:
                    param2 = param.to(dtype=param2.dtype).clone()
                else:
                    param2 = param2.mul(self.decay)
                    param2 = param2.add(param.to(dtype=param2.dtype), alpha=1-self.decay)
                state_dict[key] = param2
            self.model2.load_state_dict(state_dict, strict=False)
            self.num_updates = self.num_updates + 1
            
    def _get_decay(self, args):
        r = args.end_decay - args.decay
        p = 1 - self.num_updates/args.anneal_end_step
        return args.end_decay - r*p
    
    def save_pretrained_model(self, ckpt_dir, ckpt_name, best=False):
        if best:
            ckpt_name = 'BEST_'+ckpt_name
        self.model.save_pretrained(save_directory=os.path.join(ckpt_dir, "{}_bert.ckpt".format(ckpt_name)))
   

@dataclass
class MultimodalMRBertPretrainOutput:
    loss: float
    last_hidden_state: torch.Tensor
    hidden_states: torch.Tensor
    pooler_output: torch.Tensor
    tgt_last_hidden_state: Optional[torch.Tensor]
    tgt_hidden_states: Optional[torch.Tensor]
    tgt_pooler_output: Optional[torch.Tensor] 
    


if __name__ == '__main__':
    bert = BertModelHC.from_pretrained('bert-base-uncased')
    print(bert)
    exit()
    data = pd.read_csv('Demo_Ins0_final.csv', low_memory=False)
    data = data.iloc[:,1:]
    
    data = data.drop(columns=['ICD10_0', 'ICD10_1', 'ICD10_2', 'ICD10_3', 'ICD10_4', 'ICD10_5', 'ICD10_6', 'ICD10_7', 'ICD10_8', 'ICD10_9', 'ICD10_10', 'ICD10_11', 'ICD10_12', 'ICD10_13', 'ICD10_14', 'ICD10_15', 'ICD10_16', 'ICD10_17', 'ICD10_18', 'ICD10_19', 'ICD10_20', 'ICD10_21', 'ICD10_22', 'ICD10_23', 'ICD10_24', 'ICD10_25', 'ICD10_26', 'ICD10_27', 'ICD10_28', 'ICD10_29', 'ICD10_30', 'ICD10_31', 'ICD10_32', 'ICD10_33', 'ICD10_34', 'ICD10_35', 'ICD10_36', 'ICD10_37', 'ICD10_38', 'ICD10_39', 'ICD10_40', 'ICD10_41', 'ICD10_42', 'ICD10_43', 'ICD10_44', 'ICD10_45', 'ICD10_46', 'ICD10_47', 'ICD10_48', 'ICD10_49', 'ICD10_50', 'ICD10_51', 'ICD10_52', 'ICD10_53', 'ICD10_54', 'ICD10_55', 'ICD10_56', 'ICD10_57', 'ICD10_58', 'ICD10_59', 'ICD10_60', 'ICD10_61', 'ICD10_62', 'ICD10_63', 'ICD10_64', 'ICD10_65', 'ICD10_66', 'ICD10_67', 'ICD10_68', 'ICD10_69', 'ICD10_70', 'ICD10_71', 'ICD10_72', 'ICD10_73', 'ICD10_74', 'ICD10_75', 'ICD10_76', 'ICD10_77', 'ICD10_78'])
    
    print(data.values, data.values.shape)
    data = data.fillna(-999)
    print(data.values, data.values.shape)
    data = torch.tensor(data.values, dtype=torch.float32)
    print(data.shape)
    data = data.unsqueeze(dim=2)
    print(data.shape)
    data = data[:20,:,:]
    print(data.shape, data.dtype)
    emb = HealthFeatureEmbeddings(1, 400)
    out = emb(data)
    print(out, out.shape)