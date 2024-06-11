from models.BioT5 import *


from transformers import T5Tokenizer, AutoTokenizer, BioGptTokenizer

SUPPORTED_Model = {
    "Biot5": BioT5
}

SUPPORTED_Tokenizer = {
    "Biot5": T5Tokenizer
}

ckpt_folder = "SLM4DDI/ckpts/"

SUPPORTED_CKPT = {
    "Biot5":ckpt_folder+"text_ckpts/biot5"
}
