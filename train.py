# IMPORT PACKAGES
import torch
from torch import nn, optim

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from torch.utils.data import random_split, DataLoader

from dataset import BiLingualDataset

from model import build_transformer

from tqdm import tqdm

from config import get_config


# get sentences from the ds -> sentence by sentecne
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]
        
        

# build the tokenizer if path not there, else just load it
def get_or_build_tokenizer(ds, lang):
    
    # get the tokenizer path
    tokenizer_path = Path(f"{lang}_tokenizer.json")

    
    if not tokenizer_path.exists():
        
        # build the tokenizer
        tokenizer= Tokenizer(WordLevel(unk_token= '[UNK]'))
        tokenizer.pre_tokenizer= Whitespace()
        
        trainer= WordLevelTrainer(special_tokens= ['[SOS]', '[EOS]', '[UNK]', '[PAD]'], min_frequency= 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer= trainer)
        
        tokenizer.save(str(tokenizer_path))
        
        
    else:
        
        # load the tokenizer
        tokenizer= Tokenizer.from_file(str(tokenizer_path))
        
        
    return tokenizer



# get the dataset
def get_ds(config):
    
    # load tokenizer
    raw_ds= load_dataset("cfilt/iitb-english-hindi", split= "train")
    ds = raw_ds.select(range(200))

    
    # get the tokenizers
    src_tokenizer= get_or_build_tokenizer(ds, lang= config["src_lang"])
    trg_tokenizer= get_or_build_tokenizer(ds, lang= config["trg_lang"])
    
    
    # train test split
    train_ds_size= int(0.95 * len(ds))
    val_ds_size= len(ds) - train_ds_size
    
    train_ds, val_ds= random_split(ds, [train_ds_size, val_ds_size])
    
    
    # load the datasets
    train_ds= BiLingualDataset(
        train_ds,
        src_tokenizer,
        trg_tokenizer,
        config["src_lang"],
        config["trg_lang"],
        config["seq_len"]
    )
    
    
    val_ds= BiLingualDataset(
        val_ds,
        src_tokenizer,
        trg_tokenizer,
        config["src_lang"],
        config["trg_lang"],
        config["seq_len"]
    )
    
    
    
    # get the max seq lens for the src and trg texts
    max_src_seq_len= 0
    max_trg_seq_len= 0
    
    for item in ds:
        src_ids= src_tokenizer.encode(item["translation"][config["src_lang"]]).ids
        trg_ids= trg_tokenizer.encode(item["translation"][config["trg_lang"]]).ids
        
        max_src_seq_len= max(max_src_seq_len, len(src_ids))
        max_trg_seq_len= max(max_trg_seq_len, len(trg_ids))
        
        
    print(f"Max src seq len: {max_src_seq_len}")
    print(f"Max trg seq len: {max_trg_seq_len}")
    
    
    # load the data loaders
    train_loader= DataLoader(train_ds, config["batch_size"], shuffle= True, pin_memory= config["pin_memory"])
    val_loader= DataLoader(val_ds, 1, shuffle= True, pin_memory= config["pin_memory"])
    
    
    return train_loader, val_loader, src_tokenizer, trg_tokenizer
    
    
    
    
# get the model
def get_model(config, src_vocab_size: int, trg_vocab_size: int):
    
    # just built the transformer
    model= build_transformer(
        src_vocab_size,
        trg_vocab_size,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
        config["num_heads"],
        config["num_enc_dec_blocks"],
        config["dropout_rate"]
    )
    
    return model


# train the model
def train_model(config):
    
    # store the best epoch, min valid loss, train and valid losses
    best_epoch= 0
    min_valid_loss= float("inf")
    train_losses= []
    val_losses= []
    
    # get the data and the model
    train_loader, val_loader, src_tokenizer, trg_tokenizer= get_ds(config)
    model= get_model(config, src_tokenizer.get_vocab_size(), trg_tokenizer.get_vocab_size()).to(config["device"])
    
    # get the optim and loss
    optimizer= optim.Adam(model.parameters(), lr= config["lr"])
    loss_func= nn.CrossEntropyLoss(ignore_index= src_tokenizer.token_to_id('[PAD]'))
    
    
    # run the training loop
    for epoch in range(config["num_epochs"]):
        
        print(f"\nRunning epoch [{epoch + 1}/{config["num_epochs"]}]:-\n\n")

        # training loop
        model.train()
        
        total_train_loss= 0

        train_loop= tqdm(train_loader, desc= "Training", total= len(train_loader))

        batch_id= 0
        
        for batch in train_loop:
            
            # get the inputs
            enc_input= batch["enc_input"].to(config["device"]) # (B, S)
            dec_input= batch["dec_input"].to(config["device"]) # (B, S)
            
            encoder_mask= batch["encoder_mask"].to(config["device"]) # (B, 1, 1, S)
            decoder_mask= batch["decoder_mask"].to(config["device"]) # (B, 1, S, S)
            
            src_txt= batch["src_txt"]
            trg_txt= batch["trg_txt"]
            
            
            # pass through the transformer
            enc_output= model.encode(enc_input, encoder_mask) # (B, S, d_model)
            dec_output= model.decode(dec_input, enc_output, decoder_mask, encoder_mask) # (B, S, d_model)
            proj_output= model.project(dec_output) # (B, S, trg_vocab_size)
            
            label= batch["label"].to(config["device"]) # (B, S)
            
            # calc loss:= (B * S, trg_vocab_size) & (B * S)
            loss= loss_func(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))
            
            # update the params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # display
            train_loop.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

            if (batch_id + 1) % config["TRAIN_CONSOLE_CHECKPOINT"] == 0:
                print(f"Epoch [{epoch + 1}/{config["num_epochs"]}] ~ Batch [{batch_id + 1}] -> Train Loss: {loss.item():.4f}")

                    
            total_train_loss+= loss.item()

            batch_id+= 1
            
        avg_train_loss= total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        
        
        # validation loop
        model.eval()
        
        total_valid_loss= 0

        valid_loop= tqdm(val_loader, desc= "Validation", total= len(val_loader))

        batch_id= 0

        for batch in valid_loop:
            
            # get the inputs
            enc_input= batch["enc_input"].to(config["device"]) # (B, S)
            dec_input= batch["dec_input"].to(config["device"]) # (B, S)
            
            encoder_mask= batch["encoder_mask"].to(config["device"]) # (B, 1, 1, S)
            decoder_mask= batch["decoder_mask"].to(config["device"]) # (B, 1, S, S)
            
            src_txt= batch["src_txt"]
            trg_txt= batch["trg_txt"]
            
            # pass through the transformer
            with torch.no_grad():
                enc_output= model.encode(enc_input, encoder_mask) # (B, S, d_model)
                dec_output= model.decode(dec_input, enc_output, decoder_mask, encoder_mask) # (B, S, d_model)
                proj_output= model.project(dec_output) # (B, S, trg_vocab_size)
                
                label= batch["label"].to(config["device"]) # (B, S)
                
                
            # display
            valid_loop.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

            if (batch_id + 1) % config["VALID_CONSOLE_CHECKPOINT"] == 0:
                print(f"Epoch [{epoch + 1}/{config["num_epochs"]}] ~ Batch [{batch_id + 1}] -> Valid Loss: {loss.item():.4f}")

                    
            total_valid_loss+= loss.item()

            batch_id+= 1
            
        avg_valid_loss= total_valid_loss / len(val_loader)
        val_losses.append(avg_valid_loss)
        
        # save best weights
        if avg_valid_loss < min_valid_loss:
            min_valid_loss= avg_valid_loss
            best_epoch= epoch
            torch.save(model.state_dict(), config["best_weights_file_path"])  # Save model weights
            
        print(f"\n\nEpoch [{epoch + 1}/{config["num_epochs"]}] -> Avg Train Loss: {avg_train_loss:.4f} ~ Avg Valid Loss: {avg_valid_loss:.4f}\n\n")
                
                
        # load best weights if its the last epoch
        if (epoch + 1) == config["num_epochs"]:

            model= get_model(config, src_tokenizer.get_vocab_size(), trg_tokenizer.get_vocab_size()).to(config["device"])
            
            model.load_state_dict(torch.load(config["best_weights_file_path"]))  # Load saved weights
            model= model.to(config["device"])

            print(f"\n\nTraining done, loading the best weights...")
            
        
        
if __name__ == "__main__":
    
    model_config= get_config()
    train_model(model_config)