import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logging, huffman_encoding

class InverseResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(InverseResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, inverse=False):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        layer_class = InverseResidualLayer if inverse else ResidualLayer
        self.stack = nn.ModuleList(
            [layer_class(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
            x = F.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_conv1 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=0) 
        self.initial_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1) 
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 

    def forward(self, x):
        x = x.view(-1, 768, 1, 1).float() 
        x = F.relu(self.initial_conv1(x)) 
        x = F.relu(self.initial_conv2(x)) 
        x = self.residual_stack(x)  
        x = self.pre_quant_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) 
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=False)
        self.final_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) 
        self.final_conv2 = nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.post_quant_conv(x) 
        x = self.residual_stack(x) 
        x = F.relu(self.final_conv1(x))
        x = self.final_conv2(x) 
        x = x.view(-1, 768) 
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, show, device):
        """
        n_e: codebook size (number of codewords, default:8)
        e_dim: dimension of each codeword (default:16)
        beta: coefficient of commitment loss (default:0.25)
        """
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.show = show
        self.device = device
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0) 

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        dis = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(dis, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous(); z = z.permute(0, 3, 1, 2).contiguous()
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = codebook_loss + self.beta * commitment_loss
        z_q = z + (z_q - z).detach()
        # huffman coding
        all_indices = [int(e) for e in min_encoding_indices.view(-1)]
        huff_coded_bits = huffman_encoding(all_indices)
        bitstream_length = len(huff_coded_bits)
        if self.show:
            logging.info("Average bit length: %d bits", int(bitstream_length / z_q.shape[0]))  # bitstream length per image
        return loss, z_q


class feature_compression(nn.Module):
    def __init__(self, args, device):
        super(feature_compression, self).__init__()
        logging.info("PQVAE-shared")
        self.device = device        
        self.alpha = args.alpha
        self.beta = args.beta
        self.show_bits = args.show_bits

        self.e_dim = args.e_dim
        self.n_e = args.n_e    

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.show_bits, self.device)
        self.tanh = nn.Tanh()

    def forward(self, x):
        feature_normed = F.normalize(x, p=2, dim=1).float()
        
        encoded_x = self.encoder(feature_normed)
        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()

        quantize_loss, z_q_reconstructed = self.quantize(encoded_x)

        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)
        
        if self.training:
            similarity_matrix = torch.mm(feature_normed, decoded_x.t())
            cosine_loss = torch.mean(1-torch.diag(similarity_matrix)) # mean within a single batch
            total_loss = self.alpha * cosine_loss + quantize_loss
            return decoded_x, total_loss

        else: 
            with torch.no_grad():
                return decoded_x


