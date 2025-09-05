# import packages
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from tokenizers import Tokenizer



def causal_mask(seq_len: int):
    
    mask= torch.tril(torch.ones((1, seq_len, seq_len))).type(torch.int)
    return (mask != 0)

# my dataset
class BiLingualDataset(Dataset):
    
    def __init__(self, ds, src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer, src_lang: str, trg_lang: str, seq_len: int):
        super().__init__()
        
        self.ds= ds
        
        self.src_tokenizer= src_tokenizer
        self.trg_tokenizer= trg_tokenizer
        
        self.src_lang= src_lang
        self.trg_lang= trg_lang
        
        self.seq_len= seq_len
        
        self.sos_token= torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype= torch.int64)
        self.eos_token= torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype= torch.int64)
        self.pad_token= torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype= torch.int64)
        
        
    def __len__(self):
        
        return len(self.ds)
        
        
        
    def __getitem__(self, index):
    
        # get the src trg pair -> src and trg text
        src_trg_pair= self.ds[index]
        
        src_txt= src_trg_pair["translation"][self.src_lang]
        trg_txt= src_trg_pair["translation"][self.trg_lang]
        
        
        # get the enc, dec input tokens
        enc_input_tokens= self.src_tokenizer.encode(src_txt).ids
        dec_input_tokens= self.trg_tokenizer.encode(trg_txt).ids
        
        # get the num of padding tokens
        enc_num_padding_tokens= self.seq_len - len(enc_input_tokens) - 2 # ignoring [SOS], [EOS]
        dec_num_padding_tokens= self.seq_len - len(dec_input_tokens) - 1 # ignoring [SOS]
        
        
        # create the encoder input:= SOS -> input tokens -> EOS -> padding tokens
        enc_input= torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype= torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens)
        ])
        
        # create the decoder input:= SOS -> input tokens -> padding tokens
        dec_input= torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype= torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens)
        ])
        
        # create the decoder label:= input tokens -> EOS -> padding tokens
        dec_label= torch.cat([
            torch.tensor(dec_input_tokens, dtype= torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens)
        ])
        
        
        
        return {
            "enc_input": enc_input, # (S)
            "dec_input": dec_input, # (S)
            "label": dec_label, # (S)
            
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, S)
            "decoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.shape[0]), # (1, 1, S) & (1, S, S) -> (1, S, S)
            
            "src_txt": src_txt,
            "trg_txt": trg_txt
        }