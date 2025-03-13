from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange, repeat
from lcasr.components.batchrenorm import BatchRenorm1d
from torch.distributions import Normal
import random
from typing import Tuple, Callable, Dict
from einops.layers.torch import Rearrange
from lcasr.utils.augmentation import SpecAugment 

policy_dict = {}

class base(Module):
    def __init__(self) -> None:
        super().__init__()

class SwiGlu(base):
    def __init__(
        self,
        input_dim:int,
        output_dim:int=None,
        expansion_factor:int=2,
        ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim

        self.in_layer = nn.Linear(input_dim, input_dim*expansion_factor*2)
        self.out_layer = nn.Linear(input_dim*expansion_factor, output_dim)
        self.act = nn.SiLU()
        
    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=-1)
        c = a * self.act(b)
        return self.out_layer(c)
    

class GatedConv1d(base):
    def __init__(
        self,
        input_dim:int,
        output_dim:int=None,
        expansion_factor:int=1,
        kernel_size:Tuple[int]=(1,1),
        stride:Tuple[int]=(1,1),
        padding:Tuple[int]=(0,0),
        dropout:float=0.0
        ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim

        self.in_layer = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*expansion_factor*2, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.out_layer = nn.Conv1d(in_channels=input_dim*expansion_factor, out_channels=output_dim, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=1)
        c = a * torch.nn.functional.silu(b)
        return self.out_layer(self.dropout(c))
    



class ResidualBlock(base):
    def __init__(
            self,
            module:nn.Module,
            ) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)

class DepthWiseSeparableConv1d(base):
    def __init__(
            self,
            input_dim:int,
            output_dim:int=None,
            kernel_size:int=1,
            stride:int=1,
            padding:int=0,
            ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim
        self.depthwise = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_dim)
        self.pointwise = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
    

    def forward(self, x:Tensor):
        return self.pointwise(self.depthwise(x))



class Policy(base):
    def __init__(self) -> None:
        super().__init__()
        if type(self) == Policy: raise Exception('Policy class is not meant to be instantiated directly. Use a subclass instead!')

    def augment(self, audio):
        raise NotImplementedError
    
class NoAugmentationPolicy(Policy):
    def augment(self, audio, *args, **kwargs):
        return audio, torch.zeros_like(audio)

class FrequencyMaskingRanker(Policy):
    def __init__(self, zero_masking=True) -> None:
        super().__init__()
        self.masker = SpecAugment(n_time_masks=0, n_freq_masks=6, freq_mask_param=34, zero_masking=True)
        self.zero_masking = zero_masking


    def apply_mask(self, audio, mask):
        if not self.zero_masking:
            audio = audio * mask + (1 - mask) * audio.mean(dim=(1,2), keepdim=True)
        else:
            audio = audio * mask
        return audio
        
    def augment(self, audio, use_random=False, repeats=1):
        assert audio.dim() == 3, 'audio must be 3D tensor'
        
        if use_random:
            mask_spec = self.masker(torch.ones_like(audio))
            
            audio = self.apply_mask(audio, mask_spec)
            
            assert mask_spec.shape[1] == 80, 'mask_spec must have 80 channels'

            return audio, mask_spec[:,:,0] # mask is same across all time steps so we can just return the first one
        else: 
            return self.learnt_augmentation(audio, repeats=repeats)
        
    def learnt_augmentation(self, audio, repeats=1):
        raise NotImplementedError # must be implemented in subclass
        
class TrainableFrequencyMaskingRanker(FrequencyMaskingRanker):
    def __init__(self, zero_masking=True) -> None:
        super().__init__(zero_masking)
        if type(self) == TrainableFrequencyMaskingRanker: raise Exception('TrainableFrequencyMaskingRanker class is not meant to be instantiated directly. Use a subclass like FrequencyMaskingRanker or UnconditionalFrequencyMaskingRanker')

    def forward_pass(self, batch, device, **kwargs):
        network_inputs = {}
        network_inputs['mask'] = batch['masks'].to(device, dtype=torch.float32).squeeze(1) # B, 1, C -> B, C
        
        rewards = batch['rewards'].to(device)

        if 'audio' in batch: network_inputs['audio'] = batch['audio'].to(device)
        if 'lengths' in batch: network_inputs['lengths'] = batch['lengths'].to(device)
        if 'counts' in batch: network_inputs['counts'] = batch['counts'].to(device)

        score = self(**network_inputs)

        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(score, rewards, reduction='mean')
        elif self.loss_type == 'mult':
            loss = -torch.mean(score * rewards)
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        
        return loss, {'loss':loss}

class UnconditionalFrequencyMaskingRanker(TrainableFrequencyMaskingRanker):
    def __init__(self, zero_masking=True, loss_type='mse') -> None:
        super().__init__(zero_masking)
        assert loss_type in ['mse', 'mult']
        self.loss_type = loss_type
        self.network = nn.Sequential(
            SwiGlu(input_dim=80, output_dim=1, expansion_factor=3),
            Rearrange('b 1 -> b') 
        )


    def learnt_augmentation(self, audio, repeats=1):
        b, c, t = audio.shape
        assert c == 80, 'audio must have 80 channels'
        masks = torch.ones(b * repeats, c, 1).to(audio.device)
        masks = self.masker(masks).squeeze(-1)
        mask_scores = self(masks)
        masks = rearrange(masks, '(b r) c -> b r c', r=repeats)
        masks_scores = rearrange(mask_scores, '(b r) -> b r', r=repeats)
        # select repeat with highest score
        best_repeat = masks_scores.argmax(dim=1)
        best_masks = masks[torch.arange(b), best_repeat]
        audio = self.apply_mask(audio, best_masks.unsqueeze(-1))
        return audio, best_masks

    def forward(self, mask, **kwargs):
        return self.network(mask)




class VAEBase(base):
    def __init__(self) -> None:
        super().__init__()
        if type(self) == VAEBase: raise Exception('VAEBase class is not meant to be instantiated directly. Use a subclass like VariationalAutoEncoder')

    def reparameterize(self, mu, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if eps is None else eps
        return mu + eps * std, eps
    
    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(device)
        lengths = batch['lengths'].to(device)
        audio_out, mu, logvar = self(audio, lengths)
        
        if lengths.min() != lengths.max():
            mask = torch.arange(audio.size(-1)).to(lengths.device) < lengths[:, None]
            audio = torch.masked_fill(audio, ~mask.unsqueeze(1), 0)

        loss = nn.functional.mse_loss(audio_out, audio, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / (logvar.shape[0] * logvar.shape[1] * logvar.shape[2])
        loss += self.kld_weight * KLD
        return loss, {'loss':loss, 'KLD':KLD}, audio_out
    

class VariationalAutoEncoder(VAEBase):
    def __init__(self, input_dim=80, hidden_dim=256, latent_dim=16, layers=5, kld_weight=0.01   ):
        super().__init__()
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ) 
                for _ in range(layers)
            ],
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        self.mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        self.logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU()
                ) 
                for _ in range(layers)
            ],
        )
        self.output = nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1)

    def encode(self, x, lengths=None, eps=None, counts=None):
        in_length = x.size(-1)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        if counts is not None: # when we want to produce multiple samples from the same input
            mu = torch.repeat_interleave(mu, counts, dim=0)
            logvar = torch.repeat_interleave(logvar, counts, dim=0)

        z, eps = self.reparameterize(mu, logvar, eps)

        if lengths is not None:
            if counts is not None: lengths = torch.repeat_interleave(lengths, counts, dim=0)
            out_length = x.size(-1)
            downsampled_lengths = (lengths * out_length + in_length - 1) // in_length
            mask = torch.arange(out_length).unsqueeze(0).to(downsampled_lengths.device) < downsampled_lengths.unsqueeze(-1)
            z = torch.masked_fill(z, ~mask.unsqueeze(1), 0)
            mu = torch.masked_fill(mu, ~mask.unsqueeze(1), 0)
            logvar = torch.masked_fill(logvar, ~mask.unsqueeze(1), 0)

        return z, mu, logvar, eps



    def forward(self, x, lengths, eps=None, counts=None):
        in_length = x.size(-1)
        
        z, mu, logvar, eps = self.encode(x, lengths, eps, counts)

        x = self.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        x = self.output(x)
        return x, mu, logvar
    

class SingleStateVariationalAutoEncoder(VAEBase):
    def __init__(self, input_dim=80, hidden_dim=128, latent_dim=256, layers=6, kld_weight=0.000002, min_input_size=256):
        super().__init__()
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.min_input_size = min_input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ) 
                for _ in range(layers)
            ],
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        self.mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        self.logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)

        self.lstm_enc = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.lst_dec = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True, bidirectional=False)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU()
                ) 
                for _ in range(layers)
            ],
        )
        self.output = nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1)


    
    def bottleneck(self, x, lengths, mode='autoenc'):
        t = x.size(-1)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm_enc(x)
      
        h_n = rearrange(h_n[-1], 'b c -> b 1 c')

        if mode == 'compress':
            return h_n

        h_n = repeat(h_n, 'b 1 c -> b t c', t=t)
    
        x = torch.nn.utils.rnn.pack_padded_sequence(h_n, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lst_dec(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=t)
        x = rearrange(x, 'b t c -> b c t')
        return x


    def forward(self, x, lengths=None, mode='autoenc', eps=None):
        assert mode in ['autoenc', 'compress'], 'mode must be either autoenc or compress. compress returns the latent representation without decoding'
        in_length = x.size(-1)
        if lengths == None: lengths = torch.tensor([in_length] * x.size(0)).to(x.device)

        pad = 0
        if in_length < self.min_input_size:
            pad = self.min_input_size - in_length
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), pad).to(x.device)), dim=-1)

        x = self.encoder(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        z,_ = self.reparameterize(mu, logvar, eps)

        out_length = x.size(-1)
        downsampled_lengths = (lengths * out_length + in_length - 1) // in_length
        if lengths is not None:
            mask = torch.arange(out_length).unsqueeze(0).to(downsampled_lengths.device) < downsampled_lengths.unsqueeze(-1)
            z = torch.masked_fill(z, ~mask.unsqueeze(1), 0)
            mu = torch.masked_fill(mu, ~mask.unsqueeze(1), 0)
            logvar = torch.masked_fill(logvar, ~mask.unsqueeze(1), 0)
        
        z = self.bottleneck(z, downsampled_lengths, mode=mode)
        
        if mode == 'compress': return z

        x = self.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        x = self.output(x)
        if pad > 0: x = x[:,:,:-pad]
        return x, mu, logvar
    

class ConditionalFrequencyMaskingRanker(TrainableFrequencyMaskingRanker):
    def __init__(
            self,
            latent_dim=256,
            mel_bins=80,
            vae_config:Dict={},
            vae_state_dict_path:str=None,
            zero_masking=True, 
            loss_type='mse'
        ) -> None:
        super().__init__(zero_masking)
        assert loss_type in ['mse', 'mult']
        self.loss_type = loss_type
        self.mel_bins = mel_bins
        
        self.vae = SingleStateVariationalAutoEncoder(**vae_config)

        if vae_state_dict_path is not None: 
            vae_state_dict = torch.load(vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.vae.load_state_dict(vae_state_dict)
            print(f'Loaded VAE state dict from {vae_state_dict_path}')

        self.encode_mask = SwiGlu(self.mel_bins, output_dim=self.mel_bins, expansion_factor=1)
        self.encode_audio = nn.Sequential(
            SwiGlu(input_dim=latent_dim, output_dim=latent_dim, expansion_factor=1),
            nn.LayerNorm(latent_dim)
        )
        
        self.predict = nn.Sequential(
            SwiGlu(input_dim=latent_dim + self.mel_bins, output_dim=1, expansion_factor=2),
            Rearrange('b 1 -> b')
        )


    def learnt_augmentation(self, audio, repeats=1, lengths=None):
        b, c, t = audio.shape
        assert c == self.mel_bins, f'audio must have {self.mel_bins} channels'
        masks = torch.ones(b * repeats, c, 1).to(audio.device)

        masks = self.masker(masks).squeeze(-1)
        mask_scores = self(audio.repeat(repeats, 1, 1), masks, lengths=lengths)
        masks = rearrange(masks, '(b r) c -> b r c', r=repeats)
        masks_scores = rearrange(mask_scores, '(b r) -> b r', r=repeats)
        # select repeat with highest score
        best_repeat = masks_scores.argmax(dim=1)
        best_masks = masks[torch.arange(b), best_repeat]
        audio = self.apply_mask(audio, best_masks.unsqueeze(-1))
        
        return audio, best_masks

    def forward(self, audio, mask, lengths=None, counts=None):
        with torch.no_grad(): z_audio = rearrange(self.vae(audio, lengths=lengths, mode='compress'), 'b 1 c -> b c')

        z_audio = self.encode_audio(z_audio)
        z_mask = self.encode_mask(mask)
     

        if counts is not None: z_audio = torch.repeat_interleave(z_audio, counts, dim=0)
        
        z = torch.cat((z_audio, z_mask), dim=-1)
        return self.predict(z)


class GenerativePolicy(Policy):
    def __init__(            
            self,
            mel_bins=80,
            output_dim=80,
            hidden_dim=256,
            vae_config:Dict={},
            vae_state_dict_path:str=None,
            min_input_size=160
        ) -> None:
        super().__init__()

        self.vae = VariationalAutoEncoder(**vae_config)

        if vae_state_dict_path is not None:
            vae_state_dict = torch.load(vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.vae.load_state_dict(vae_state_dict)
            print(f'Loaded VAE state dict from {vae_state_dict_path}')

        self.output = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1)
        self.hidden_dim = hidden_dim
        self.mel_bins = mel_bins
        self.output_dim = output_dim
        self.min_input_size = min_input_size
        self.eps_clip = 0.2 # default value for PPO

    def forward(self, x, lengths, eps=None, counts=None):
        if x.size(-1) < self.min_input_size:
            pad = self.min_input_size - x.size(-1)
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), pad).to(x.device)), dim=-1)
         
        else: pad = 0
        
        in_length = x.size(-1)

        with torch.no_grad():
            z, mu, logvar, eps = self.vae.encode(x, lengths, eps, counts)

        x = self.vae.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        if pad > 0: x = x[:,:,:-pad]
        x = self.output(x)
   
        return x, eps
    

class ConditionalMaskingPolicy(GenerativePolicy):
    @torch.no_grad()
    def augment(self, audio, eps=None, lengths=None):
        predictions, eps = self(audio, lengths, eps)
        probs = predictions.sigmoid() # 0 = mask, 1 = no mask
        mask = torch.bernoulli(probs)
        audio = audio * mask

        return audio, mask, {'probs':probs, 'eps':eps}
    
    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(dtype=torch.float32, device=device)
        lengths = batch['lengths'].to(device)
        #ds_lengths = batch['ds_lengths'].to(device)
        masks = batch['masks'].to(device, dtype=torch.long).squeeze(1)
        rewards = batch['rewards'].to(device)
        old_probs = batch['probs'].to(dtype=torch.float32, device=device).squeeze(1)
        eps = batch['eps'].to(dtype=torch.float32, device=device).squeeze(1)
        counts = batch['counts'].to(device)
        
        out, _ = self(audio, lengths, eps=eps, counts=counts)
        probs = out.sigmoid()
        
        p_of_mask = probs * masks + (1 - probs) * (1 - masks)
        log_p_of_mask = torch.log(p_of_mask + 1e-8)


        old_p_of_mask = old_probs * masks + (1 - old_probs) * (1 - masks)
        old_log_p_of_mask = torch.log(old_p_of_mask + 1e-8)
        
        ratios = torch.exp(log_p_of_mask - old_log_p_of_mask)

        surr1 = ratios * rewards[:,None,None]
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * rewards[:,None,None]

        loss = -torch.min(surr1, surr2) 

        lengths = lengths.repeat_interleave(counts, dim=0)
        pad_mask = torch.arange(audio.size(-1)).to(lengths.device) >= lengths[:, None]
        loss = torch.masked_fill(loss, pad_mask.unsqueeze(1), 0)

        loss = loss.sum() / (~pad_mask).sum()
      
        return loss, {'loss':loss}



class AdditivePolicy(Policy):
    def __init__(
            self,
            input_dim=80,
            hidden_dim=80,
            output_dim=20,
            expansion_factor=2,
        ) -> None:
        super().__init__()

        hidden_dim = hidden_dim
        downsample_kernel = 7
        gated_kernel = (9,9)
        
        self.downsample_kernel = downsample_kernel
        self.gated_kernel = gated_kernel
        self.input_dim = input_dim
        self.block_layers = 5
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encode = (
                nn.Sequential(
                    *[
                        ResidualBlock(nn.Sequential(
                            GatedConv1d(
                                input_dim=hidden_dim, 
                                output_dim=hidden_dim, 
                                expansion_factor=expansion_factor, 
                                kernel_size=gated_kernel, 
                                stride=(1,1), 
                                padding=("same", "same"),
                            ),
                            BatchRenorm1d(hidden_dim) 
                        ))   for _ in range(self.block_layers)
                    ],
                    Rearrange('b c t -> b t c'),
                    nn.Linear(hidden_dim,hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim,output_dim*input_dim),
            ))
    

    def forward(self, x):
        x = self.encode(x)
        return x    
    
    def augment(self, audio, sample=True, return_probs=False, use_random=False, lengths=None):

        if not use_random:
            pred = self(audio) # b, c, t
            pred = rearrange(pred, 'b t (c p) -> b t c p', p=self.output_dim)
            pred = pred.softmax(-1)
        else:
            b, c, t = audio.shape
            pred = (torch.zeros(b, t, self.input_dim, self.output_dim) + 1/self.output_dim).to(audio.device)
        

        b = pred.shape[0]
        pred = rearrange(pred, 'b t c p-> (b t c) p')

        
        if sample: indexes = torch.multinomial(pred, 1).squeeze(-1)
        else: indexes = pred.argmax(-1)
        if return_probs: probs = torch.gather(pred, -1, indexes.unsqueeze(-1)).squeeze(-1)
        
        values = torch.round(-0.5 + indexes.float() * (2.0 / (self.output_dim - 1)), decimals=1)
        noise = rearrange(values, '(b t c) -> b c t', b=b, c=self.input_dim)
        
        if lengths is not None and lengths.min() != lengths.max():
            b, _, t = noise.shape
            pad_mask = torch.arange(t).to(lengths.device) >= lengths[:, None]
            noise = torch.masked_fill(noise, pad_mask.unsqueeze(1), 0)

        augmented_audio = audio + noise
        
        if return_probs: return augmented_audio, noise, probs
        else: return augmented_audio, noise
    
    def discretize(self, float_mask):
        indices = torch.round((float_mask + 0.5) * ((self.output_dim - 1) / 2.0)).long()
        indices = torch.clamp(indices, 0, self.output_dim - 1)

        return indices
    

policy_dict['FrequencyMaskingRanker'] = FrequencyMaskingRanker
policy_dict['UnconditionalFrequencyMaskingRanker'] = UnconditionalFrequencyMaskingRanker
policy_dict['ConditionalFrequencyMaskingRanker'] = ConditionalFrequencyMaskingRanker

policy_dict['NoAugmentation'] = NoAugmentationPolicy
policy_dict['AdditivePolicy'] = AdditivePolicy
policy_dict['ConditionalMaskingPolicy'] = ConditionalMaskingPolicy
policy_dict['default'] = AdditivePolicy