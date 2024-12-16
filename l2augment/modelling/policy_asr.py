import torch, torch.nn as nn, torch.nn.functional as F

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange
from functools import partial
from lcasr.components import fused_dense, subsampling, convolution, decoder, wrappers
from lcasr.utils.helpers import exists
from lcasr.components.helpers import get_act
ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling
from lcasr.components.positional_encodings import RotaryPositionalEmbedding, apply_rotary, LearnableFourierPosEnc


try: from apex.normalization import FusedRMSNorm as DEFAULT_NORM, FusedRMSNorm as RMSNorm, FusedLayerNorm as LayerNorm
except: 
    from lcasr.components.normalisation import RMSNorm as RMSNorm, RMSNorm as DEFAULT_NORM
    from torch.nn import LayerNorm as LayerNorm


from lcasr.models.sconformer_xl import SCConformerXL
PreNorm, Scale = wrappers.PreNorm, wrappers.Scale

from lcasr.components.attention import Attention


# TODO: 
# -. remove caching stuff as it is not used anymore

class SCConformerXLPolicy(SCConformerXL): 
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.decoder = None
      
        self.decoder_policy = decoder.ASRLinearSCDecoder(
            d_model = kwargs.get('d_model', 256),
            vocab_size = 1,
            norm = True,
            norm_fn = LayerNorm
        )
        self.feature_combine = nn.Linear(2 * kwargs.get('d_model', 256), kwargs.get('d_model', 256))

        
    def forward(
            self, 
            audio_signal,
            masks,
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''
        length = None
        decoder = self.decoder_policy
        max_audio_length: int = audio_signal.size(-1)

        cached_kvs = None
        cached_kvs_lengths = None
      

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        mask_signal = torch.transpose(masks, 1, 2)
        
        audio_signal, length = self.subsampling(audio_signal, lengths = length) 
        mask_signal, _ = self.subsampling(mask_signal, lengths = length)
        
        audio_signal = torch.concat([audio_signal, mask_signal], dim = -1)
        audio_signal = self.feature_combine(audio_signal)

        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None
   
        full_kv_lengths = length 
        if self.use_rotary:
            max_seq_len = full_kv_lengths.max()
            q_offset = 0 if cached_kvs is None else cached_kvs.shape[1]
      
            cos, sin = self.rotary_pos_emb(max_seq_len, audio_signal.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = self.rotary_pos_emb.learned_freq)
        

        if length.max() == length.min():
            att_mask, mask = None, None
        else:
            full_kv_mask = torch.arange(full_kv_lengths.max(), device=audio_signal.device).expand(audio_signal.size(0), full_kv_lengths.max()) >= full_kv_lengths.unsqueeze(1)
            if audio_signal.device.type == 'cuda' and self.flash_attn:
                att_mask = ~full_kv_mask
            else:
                qmask, kmask = ~mask, ~full_kv_mask
                att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
                att_mask = att_mask.to(audio_signal.dtype) * -torch.finfo(audio_signal.dtype).max

        pad_mask = mask 
    
        audio_signal = self.fourier_pos_enc(audio_signal)
        
        for lth, layer in enumerate(self.layers):

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    length,
                    pad_mask, # pad_mask
                    self.flash_attn,
                    rotary_emb_fn,
                )
            else:
                audio_signal = layer(
                    x = audio_signal, 
                    attn_mask = att_mask, 
                    length = length,
                    pad_mask = pad_mask,
                    flash_attn = self.flash_attn,
                    rotary_emb_fn = rotary_emb_fn
                )
                 

        
        final_posts = decoder(x = audio_signal) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()

        return final_posts


class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        conv_kernel_size,
        dropout_ff,
        dropout_conv,
        dropout_attn,
        layer_idx,
        total_layers,
        head_dim,
        n_heads,
        default_norm = DEFAULT_NORM,
        sandwich_norm = False,
        bias_in_ff = True,
        transformer = False,
        conv_expansion_factor = 1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.trasformer = transformer

        
        if not self.trasformer:
        
            self.conv = PreNorm(
                d_model = d_model, 
                fn = ConformerConvolution(
                    d_model = d_model,
                    kernel_size = conv_kernel_size,
                    norm_type = kwargs.get('conv_norm', 'batch_renorm'),
                    exp_factor = conv_expansion_factor,
                ),
                norm = default_norm
            )
            self.do_conv = nn.Dropout(dropout_conv)

        if not self.trasformer:
            self.ff1 = Scale(0.5, PreNorm(
                d_model = d_model, 
                fn = ConformerFeedForward(
                    d_model, 
                    bias1 = bias_in_ff, 
                    bias2 = bias_in_ff,
                    checkpoint_lvl = kwargs.get('ff_checkpoint_lvl', 0)
                ), 
                norm = default_norm, 
                sandwich_norm = sandwich_norm
            ))
        
        self.ff2 = Scale(0.5, PreNorm(
            d_model = d_model, 
            fn = ConformerFeedForward(
                d_model, 
                bias1 = bias_in_ff, 
                bias2 = bias_in_ff,
                checkpoint_lvl = kwargs.get('ff_checkpoint_lvl', 0)
            ), 
            norm = default_norm, 
            sandwich_norm = sandwich_norm
        ))

        self.do_ff = nn.Dropout(dropout_ff)

        self.has_attention = kwargs.get('has_attention', True)

        if self.has_attention:
            self.attend = PreNorm(
                d_model = d_model, 
                fn = Attention(
                    n_feats = d_model,
                    head_dim = head_dim,
                    n_heads = n_heads,
                    dropout = dropout_attn,
                    bias = False,
                    layer_idx = layer_idx,
                    **kwargs
                ),
                norm = default_norm,
            )
            self.attn_norm_out = default_norm(d_model) if sandwich_norm else lambda x: x
            self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large

        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, length, pad_mask, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        if not self.trasformer:
            x = self.do_ff(self.ff1(x)) + x

        if self.has_attention:
            x = self.attn_norm_out(self.do_attn_out(self.attend(
                x = x,
                attn_mask = attn_mask,
                length = length,
                pad_mask = pad_mask,
                flash_attn = flash_attn,
                rotary_emb_fn = rotary_emb_fn
            ))) + x
        
        if not self.trasformer:
            x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)
        return x




if __name__ == '__main__':
    # run test
    model = SCConformerXL(vocab_size=4096, head_dim=256, n_heads=3, attention_window_size=128)
    audio = torch.randn(2, 80, 1000)
    lengths = torch.tensor([1000, 500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    audio = audio.to(device)
    lengths = lengths.to(device)
    out = model(audio, length=lengths)
    print(out['final_posteriors'].shape)
    