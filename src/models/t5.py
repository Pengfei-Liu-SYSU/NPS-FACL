import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
#from models.base_models import MolEncoder, TextEncoder

class BioT5(nn.Module):
    def __init__(self, config = None):
        super(BioT5, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/root/autodl-tmp/NPS_classification/ckpts/biot5+"
            }
        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/biot5+"
        self.main_model = T5ForConditionalGeneration.from_pretrained(self.config.ckpt)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.ckpt)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
    
    def encode(self, input_ids, attention_mask):
        outputs = self.main_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)


class MolT5(nn.Module):
    def __init__(self, config = None):
        super(MolT5, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/root/autodl-tmp/NPS_classification/ckpts/molt5-base"
            }
        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/molt5-base"
        self.main_model = T5ForConditionalGeneration.from_pretrained(self.config.ckpt)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.ckpt)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
    
    def encode(self, input_ids, attention_mask):
        outputs = self.main_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)


class MolT5_add(nn.Module):
    def __init__(self, config = None):
        super(MolT5_add, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/root/autodl-tmp/NPS_classification/ckpts/molt5-base-add"
            }
        self.config = config
        self.config.ckpt = "/root/autodl-tmp/NPS_classification/ckpts/molt5-base-add"
        self.main_model = T5ForConditionalGeneration.from_pretrained(self.config.ckpt)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.ckpt)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
    
    def encode(self, input_ids, attention_mask):
        outputs = self.main_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs  # [batch_size, seq_len, hidden_size])

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)
