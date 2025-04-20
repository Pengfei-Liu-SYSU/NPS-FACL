from models.bert import SciBERT, ChemBERTa, SciBERT_add_tokens
from models.t5 import BioT5, MolT5, MolT5_add
from transformers import GPT2Tokenizer, AutoTokenizer, T5Tokenizer

SUPPORTED_Model = {
    "chemberta": ChemBERTa,
    "scibert": SciBERT,
    "biot5": BioT5,
    "molt5": MolT5,
    "molt5_add": MolT5_add,
    "scibert+": SciBERT_add_tokens
}


SUPPORTED_Tokenizer = {
    "chemberta": AutoTokenizer,
    "scibert": AutoTokenizer,
    "biot5": T5Tokenizer,
    "molt5": T5Tokenizer,
    "molt5_add": T5Tokenizer,
    "scibert+": AutoTokenizer
}

ckpt_folder = "/root/autodl-tmp/NPS_classification/ckpts/"

SUPPORTED_CKPT = {
    "chemberta": ckpt_folder+"chemberta",
    "scibert": ckpt_folder+"scibert",
    "biot5": ckpt_folder+"biot5+",
    "molt5": ckpt_folder+"molt5-base",
    "molt5_add": ckpt_folder+"molt5-base-add",
    "scibert+": ckpt_folder+"scibert_add_tokens"
}
