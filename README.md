# **🌍 Transformer Machine Translation (English → Hindi)**

This project implements a Transformer model from scratch for sequence-to-sequence translation using the Hugging Face Datasets and custom tokenization with tokenizers.

## 🧠 Model Overview

### ✅ Transformer Architecture

- Built from scratch using `torch.nn.Module`.
    
- Components:
    
    - **Embedding layers**
        
    - **Positional encodings**
        
    - **Multi-head self-attention**
        
    - **Feed-forward layers**
        
    - **Encoder–Decoder structure**
        
- Final projection to vocab size for predictions.
    

---

## 🗃️ Dataset

- Using `HuggingFace`'s `cfilt/iitb-english-hindi` -> https://huggingface.co/datasets/cfilt/iitb-english-hindi

    
- Each sample contains a pair:
    
    - `"translation": {"en": "source sentence", "hi": "target sentence"}`
        

---

## 🔤 Tokenization

### `build_tokenizer(ds, lang, save_path)`

- Trains a **WordLevel tokenizer** with special tokens:
    
    - `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`
        
- Saves and reuses tokenizers to avoid retraining.
    
- Language-specific (e.g., English and Hindi).
    

---

## 📦 Custom Dataset Class

### `CustomDataset(Dataset)`

- Inputs:
    
    - Dataset object
        
    - Tokenizers
        
    - Sequence length
        
- Returns:
    
    - `enc_inputs`, `dec_inputs`, `labels`
        
    - Encoder & decoder masks
        
    - Raw source and target texts
        

Handles:

- Token encoding
    
- Padding
    
- Truncation
    
- Special tokens (`[SOS]`, `[EOS]`, `[PAD]`)
    

---

## 🔁 Inference (Greedy Decoding)

### `greedy_decode(enc_inputs, enc_mask)`

- Step-by-step token generation
    
- Starts with `[SOS]` token
    
- Stops at `[EOS]` or max length
    
- Uses model’s `encode`, `decode`, and `final_projection` methods
    

---


## 🚀 Transformer Inference Loop

### `transformer_inference()`

- Loops over `test_loader`
    
- For each input:
    
    - Encodes the sentence
        
    - Performs greedy decoding
        
    - Compares predicted vs actual translation
        
- Logs:

```
SRC: I love to learn deep learning.
TRG: मुझे गहरा सीखना पसंद है।
PRD: मुझे डीप लर्निंग पसंद है।

--------------------------------------------------

SRC: Give your application an accessibility workout
TRG: अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें
PRD: अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें

--------------------------------------------------

SRC: Dogtail
TRG: श्वानपुच्छ
PRD: श्वानपुच्छ

--------------------------------------------------
```

## 📈 Future Improvements

- **Use Full Dataset**:  
    The dataset has ~1.5 million sentence pairs (`cfilt/iitb-english-hindi`), but currently only the first **15,000** samples are used due to GPU limitations.  

    ➤ To scale:
```
ds = load_dataset("cfilt/iitb-english-hindi", split="train")
# Instead of:
ds = ds.select(range(15000))
```

* **Increase Model Depth & Width**:  
The model currently uses:

- `4` encoder/decoder blocks instead of `6`
    
- `4` attention heads instead of `8`  
    These were reduced for GPU memory compatibility.  
    ➤ Update the model definition:

```
model = build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_trg.get_vocab_size(),
    final_seq_len,
    final_seq_len,
    num_heads=8,                # originally intended
    num_enc_dec_blocks=6        # originally intended
)
```

- **Implement Beam Search**:  
    Replace greedy decoding with beam search for more accurate translations.
    
- **Evaluate with BLEU or ROUGE**:  
    Add proper translation evaluation metrics for validation and comparison.
    
- **Visualize Attention Weights**:  
    Helpful for analyzing how the model attends to input tokens.
    
- **Add FP 16 Training:  
    Helps reduce GPU memory usage and speeds up training.


## 🤝 Credits

This project was built with inspiration, guidance, and knowledge from the following excellent resources:

- 📘 **Jay Alammar’s Illustrated Transformer**  
    [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)  
    A beautifully visual and intuitive explanation of the Transformer architecture.
    
- 🎥 **Umar Jamil’s Code Walkthrough – Transformer from Scratch**  
    [https://www.youtube.com/watch?v=ISNdQcPhsts](https://www.youtube.com/watch?v=ISNdQcPhsts)  
    Practical implementation guidance and architecture breakdown.
    
- 📄 **"Attention is All You Need" – Vaswani et al. (2017)**  
    [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)  
    The original paper that introduced the Transformer model.
    
- 🎓 **Deep Learning Playlist (Theory Lectures)**  
    [https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn](https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)  
    A YouTube playlist that helped in deeply understanding the theory behind Transformers.
