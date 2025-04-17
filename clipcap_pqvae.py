import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from typing import Optional
import math
from pqvae_shared import Encoder, Decoder, VectorQuantizer
from typing import Tuple

"""
"Image captioning using compressed CLIP features"
"For mapping functions, we simply use MLP instead of Transformer"
"""

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class compress_then_caption(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def __init__(self, args, device="cuda", prefix_length: int=10, prefix_size: int=768):
        super(compress_then_caption, self).__init__()
        self.device = device
        self.alpha = args.alpha
        self.beta = args.beta
        self.e_dim = args.e_dim
        self.n_e = args.n_e    
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer(self.n_e, self.e_dim, self.beta, show=False, device=self.device)

        self.prefix_length = prefix_length 
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] 
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                    self.gpt_embedding_size * prefix_length)) 

    def forward(self, tokens, x, labels: Optional[torch.Tensor] = None):  
        # compress
        feature_normed = F.normalize(x, p=2, dim=1).float()
        encoded_x = self.encoder(feature_normed)
        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()
        quantize_loss, z_q_reconstructed = self.quantize(encoded_x)
        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)

        # captioning
        prefix_projections = self.clip_project(decoded_x).view(-1, self.prefix_length, self.gpt_embedding_size)  
        embedding_text = self.gpt.transformer.wte(tokens) 
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None: 
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels)

        return out, quantize_loss

