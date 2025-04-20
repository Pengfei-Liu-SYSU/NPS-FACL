import torch
import torch.nn as nn

from models import SUPPORTED_Model
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchEncoding
import re

def pad_tensors_to_max_length(tensors, max_len, device, dim=0):
    """Pad each tensor in the list to the specified max_len."""
    padded_tensors = []
    for tensor in tensors:
        pad_size = max_len - tensor.size(dim)
        padded_tensor = torch.cat([tensor, torch.zeros(*tensor.shape[:-1], pad_size).to(device)], dim=dim)
        padded_tensors.append(padded_tensor)
    return padded_tensors


class MainModel(nn.Module):
    def __init__(self, config=None):
        super(MainModel, self).__init__()
        self.config = config
        self.config.hidden_size = 768
        self.language_model  = SUPPORTED_Model[self.config.model_name](config)

        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, mol):

        input_ids = mol["input_ids"]  # [batch_size, seq_len]
        attention_mask = mol["attention_mask"]  # [batch_size, seq_len]

        if(self.config.model_name in ['biot5', 'molt5','molt5_add'] ):
            encoder_outputs = self.language_model.main_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            encoder_outputs = self.language_model.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )


        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    

        logits = self.classification_head(cls_embedding)  # [batch_size, 1]
        logits = logits.squeeze(-1)  
    

        if "output" in mol: 
            loss = self.criterion(logits, mol["output"].float())  # [batch_size]
            return loss, logits
        else:
            return logits
    
    def predict(self, mol):

        self.eval()  
        with torch.no_grad():
            logits = self.forward(mol)  
            probability = torch.sigmoid(logits) 
            predicted_class = (probability > 0.5).long() 

        return probability, predicted_class

