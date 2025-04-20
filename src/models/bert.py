import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SciBERT(nn.Module):
    def __init__(self, config=None):
        super(SciBERT, self).__init__()

        if config is None:
            config = {
                "name": "scibert",
                "ckpt": "/root/autodl-tmp/NPS_classification/ckpts/scibert"  
            }

        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/scibert"
        self.main_model = AutoModel.from_pretrained(self.config.ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ckpt)


        if "stop_grad" in config:
            for param in self.main_model.parameters():
                param.requires_grad = False


        self.hidden_size = self.main_model.config.hidden_size  # 768
        self.output_dim = self.hidden_size

    def forward(self, input_ids, attention_mask):

        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
    def encode(self, input_ids, attention_mask):

        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])

class SciBERT_add_tokens(nn.Module):
    def __init__(self, config=None):
        super(SciBERT_add_tokens, self).__init__()

        if config is None:
            config = {
                "name": "scibert",
                "ckpt": "/root/autodl-tmp/NPS_classification/ckpts/scibert_add_tokens"  
            }

        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/scibert_add_tokens"
        self.main_model = AutoModel.from_pretrained(self.config.ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ckpt)

        if "stop_grad" in config:
            for param in self.main_model.parameters():
                param.requires_grad = False

        self.hidden_size = self.main_model.config.hidden_size  # 768
        self.output_dim = self.hidden_size

    def forward(self, input_ids, attention_mask):

        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
    def encode(self, input_ids, attention_mask):

        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])
        
class ChemBERTa(nn.Module):
    def __init__(self, config=None):
        super(ChemBERTa, self).__init__()


        if config is None:
            config = {
                "name": "chemberta",
                "ckpt": "/root/autodl-tmp/NPS_classification/ckpts/chemberta"  
            }

        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/chemberta"
        self.main_model = AutoModel.from_pretrained(self.config.ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ckpt)


        if "stop_grad" in config:
            for param in self.main_model.parameters():
                param.requires_grad = False


        self.hidden_size = self.main_model.config.hidden_size  # 768
        self.output_dim = self.hidden_size

    def forward(self, input_ids, attention_mask):

        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
    def encode(self, input_ids, attention_mask):
        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])