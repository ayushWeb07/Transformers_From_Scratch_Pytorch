# IMPORT PACKAGES
import torch
from torch import nn, optim
import math

# 1) INPUT EMBEDS
class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        
        super().__init__()
        
        self.d_model= d_model
        self.vocab_size= vocab_size
        
        self.embedding= nn.Embedding(vocab_size, d_model)
        
        
    def forward(self, x: torch.Tensor):
        out= self.embedding(x) * math.sqrt(self.d_model)
        return out



# 2) POS EMBEDS
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout_rate: float):
        
        super().__init__()
        
        
        self.d_model= d_model
        self.seq_len= seq_len
        self.dropout= nn.Dropout(p= dropout_rate)
        
        # pos. enc.
        pe= torch.zeros((seq_len, d_model)) # (S, d_model)
        indices= torch.arange(0, d_model, 2).float() # (d_model / 2)
        positions= torch.arange(0, seq_len).unsqueeze(1) # (S, 1)
        
        div_term= torch.exp(
            indices * (-math.log(10000) / d_model)
        );
        
        # fill the even spots with sin
        pe[:, 0::2]= torch.sin(positions * div_term)
        
        # fill the odd spots with cos
        pe[:, 1::2]= torch.cos(positions * div_term)
        
        # add batch dims
        pe= pe.unsqueeze(0) # (1, S, d_model)
        
        # register to buffer
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, d_model)
        x= x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
        
        
# 3) FEED FORWARD MLP
class FeedForwardMLP(nn.Module):
    
    def __init__(self, d_model: int, dropout_rate: float):
        
        super().__init__()
        
        # seq block:- 
        # (d_model, d_model * 4) -> GeLU -> Dropout -> (d_model * 4, d_model)
        
        self.mlp= nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, d_model)
        out= self.mlp(x)
        return out
    
    
    
# 4) MHA
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int):
        
        super().__init__()
        
        self.d_model= d_model
        self.n_heads= n_heads
        self.d_head= (d_model // n_heads)
        
        # query, key, value, output weight matrices
        self.w_q= nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)
        self.w_o= nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        
        # get the query, key, value -> (B, S, d_model)
        query= self.w_q(query)
        key= self.w_k(key)
        value= self.w_v(value)
        
        # change shape of query, key, value:= (B, S, d_model) -> (B, S, n_heads, d_head)
        query= query.view(query.shape[0], query.shape[1], self.n_heads, self.d_head)
        key= key.view(key.shape[0], key.shape[1], self.n_heads, self.d_head)
        value= value.view(value.shape[0], value.shape[1], self.n_heads, self.d_head)
        
        # change shape of query, key, value:= (B, S, n_heads, d_head) -> (B, n_heads, S, d_head)
        query= query.transpose(1, 2)
        key= key.transpose(1, 2)
        value= value.transpose(1, 2)
        
        # calc attention scores
        atten_scores= query @ key.transpose(-1, -2) # (B, n_heads, S, S)
        atten_scores/= math.sqrt(self.d_head)
        
        # mask if there
        if(mask is not None):
            atten_scores.masked_fill_(mask == 0, -1e9)
        
        # calc attention weights
        atten_weights= torch.softmax(atten_scores, dim= -1) # (B, n_heads, S, S)
        atten_weights= atten_weights @ value # (B, n_heads, S, d_head)
        
        
        # change shape of attention weights:= (B, n_heads, S, d_head) -> (B, S, n_heads, d_head)
        atten_weights= atten_weights.transpose(1, 2)
        
        # change shape of attention weights:= (B, S, n_heads, d_head) -> (B, S, d_model)
        atten_weights= atten_weights.contiguous().view(atten_weights.shape[0], -1, self.d_model)
        
        
        out= self.w_o(atten_weights) # (B, S, d_model)
        
        return out
        
        
        
# 5) ENCODER BLOCK
class EncoderBlock(nn.Module):
    
    def __init__(self, d_model: int, mha_block: MultiHeadAttention, mlp_block: FeedForwardMLP):
        
        super().__init__()
        
        self.mha_block= mha_block
        self.mlp_block= mlp_block
        
        self.norm1= nn.RMSNorm(d_model)
        self.norm2= nn.RMSNorm(d_model)
        
        
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        
        out= x + self.mha_block(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        out= out + self.mlp_block(self.norm2(out))
        
        return out
    
    
# 6) ENCODER
class Encoder(nn.Module):
    
    def __init__(self, encoder_blocks: nn.ModuleList, d_model: int):
        
        super().__init__()
        
        self.norm= nn.RMSNorm(d_model)
        
        self.encoder_blocks= encoder_blocks
        
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        
        # pass it through all the encoder blocks
        for block in self.encoder_blocks:
            x= block(x, src_mask)
            
        out= self.norm(x)
        
        return out
    
    
    
# 7) DECODER BLOCK
class DecoderBlock(nn.Module):
    
    def __init__(self, d_model: int, self_atten_block: MultiHeadAttention, cross_atten_block: MultiHeadAttention, mlp_block: FeedForwardMLP):
        
        super().__init__()
        
        self.self_atten_block= self_atten_block
        self.cross_atten_block= cross_atten_block
        self.mlp_block= mlp_block
        
        self.norm1= nn.RMSNorm(d_model)
        self.norm2= nn.RMSNorm(d_model)
        self.norm3= nn.RMSNorm(d_model)
        
        
    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor, dec_mask: torch.Tensor, enc_mask: torch.Tensor):
        
        out= dec_input + self.self_atten_block(self.norm1(dec_input), self.norm1(dec_input), self.norm1(dec_input), dec_mask)
        out= out + self.cross_atten_block(self.norm2(out), self.norm2(enc_output), self.norm2(enc_output), enc_mask)
        out= out + self.mlp_block(self.norm3(out))
        
        return out
    
    
# 8) DECODER
class Decoder(nn.Module):
    
    def __init__(self, decoder_blocks: nn.ModuleList, d_model: int):
        
        super().__init__()
        
        self.norm= nn.RMSNorm(d_model)
        
        self.decoder_blocks= decoder_blocks
        
    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor, dec_mask: torch.Tensor, enc_mask: torch.Tensor):
        
        # pass it through all the encoder blocks
        for block in self.decoder_blocks:
            x= block(dec_input, enc_output, dec_mask, enc_mask)
            
        out= self.norm(x)
        
        return out
    
    
    
# 9) FINAL PROJECTION LINEAR LAYER
class LinearProjection(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int):
        
        super().__init__()
        
        self.proj= nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, d_model)
        out= torch.softmax(self.proj(x), dim= -1) # (B, S, vocab_size)
        
        return out
    
    
    
# 10) TRANSFORMERS
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 encoder_token_embedding_layer: InputEmbedding, decoder_token_embedding_layer: InputEmbedding,
                 encoder_pos_embedding_layer: PositionalEmbedding, decoder_pos_embedding_layer: PositionalEmbedding,
                 linear_projection_layer: LinearProjection):
        
        super().__init__()
        
        
        self.encoder= encoder
        self.decoder= decoder
        
        self.encoder_token_embedding_layer= encoder_token_embedding_layer
        self.decoder_token_embedding_layer= decoder_token_embedding_layer
        
        self.encoder_pos_embedding_layer= encoder_pos_embedding_layer
        self.decoder_pos_embedding_layer= decoder_pos_embedding_layer
        
        self.linear_projection_layer= linear_projection_layer
        
        
    def encode(self, enc_input: torch.Tensor, enc_mask: torch.Tensor):
        
        enc_input= self.encoder_token_embedding_layer(enc_input)
        enc_input= self.encoder_pos_embedding_layer(enc_input)
        enc_output= self.encoder(enc_input, enc_mask)
        
        return enc_output
    
    
    def decode(self, dec_input: torch.Tensor, enc_output: torch.Tensor, dec_mask: torch.Tensor, enc_mask: torch.Tensor):
        
        dec_input= self.decoder_token_embedding_layer(dec_input)
        dec_input= self.decoder_pos_embedding_layer(dec_input)
        dec_output= self.decoder(dec_input, enc_output, dec_mask, enc_mask)
        
        return dec_output
        
        
        
    def project(self, x: torch.Tensor):
        
        out= self.linear_projection_layer(x)
        return out
        
        
        
        
# 11) BUILD TRANSFORMER
def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int= 512, num_heads: int= 8, num_enc_dec_blocks= 6, dropout_rate: float= 0.1):
    
    # the enc and dec input token embeds layers
    enc_input_embed_layer= InputEmbedding(d_model, src_vocab_size)
    dec_input_embed_layer= InputEmbedding(d_model, trg_vocab_size)
        

    # the enc and dec pos. embeds layers
    enc_pos_embed_layer= PositionalEmbedding(d_model, src_seq_len, dropout_rate)
    dec_pos_embed_layer= PositionalEmbedding(d_model, trg_seq_len, dropout_rate)
    
        
        
    # create the encoder
    encoder_blocks= nn.ModuleList([])
    
    for _ in range(num_enc_dec_blocks):
        mha_block= MultiHeadAttention(d_model, num_heads)
        mlp_block= FeedForwardMLP(d_model, dropout_rate)
        enc_block= EncoderBlock(d_model, mha_block, mlp_block)
        
        encoder_blocks.append(enc_block)
        
    encoder= Encoder(encoder_blocks, d_model)
    
    
    # create the decoder
    decoder_blocks= nn.ModuleList([])
    
    for _ in range(num_enc_dec_blocks):
        self_atten_block= MultiHeadAttention(d_model, num_heads)
        cross_atten_block= MultiHeadAttention(d_model, num_heads)
        
        mlp_block= FeedForwardMLP(d_model, dropout_rate)
        
        dec_block= DecoderBlock(d_model, self_atten_block, cross_atten_block, mlp_block)
        
        decoder_blocks.append(dec_block)
        
    decoder= Decoder(decoder_blocks, d_model)
    
    
    # create the final projection layer
    linear_projection_layer= LinearProjection(trg_vocab_size, d_model)
    
    # create the transformer
    transformer= Transformer(
        encoder,
        decoder,
        
        enc_input_embed_layer,
        dec_input_embed_layer,
        
        enc_pos_embed_layer,
        dec_pos_embed_layer,
        
        linear_projection_layer
    )
    
    return transformer