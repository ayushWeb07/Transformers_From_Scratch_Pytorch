import torch

def get_config():
    
    return {
        "seq_len": 350,
        "d_model": 512,
        "num_heads": 4,
        "num_enc_dec_blocks": 3,
        "dropout_rate": 0.1,
        "src_lang": "en",
        "trg_lang": "hi",
        "batch_size": 4,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "pin_memory": torch.cuda.is_available(),
        "lr": 1e-4,
        "num_epochs": 5,
        "TRAIN_CONSOLE_CHECKPOINT": 100,
        "VALID_CONSOLE_CHECKPOINT": 1,
        "best_weights_file_path": "best_weights.pth"
    }