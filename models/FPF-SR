import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dctnet import dct_channel_block
from models.MultiHeadAttention import MultiHeadAttention

class FrequencyDomainModel(nn.Module):
    def __init__(self, configs):
        super(FrequencyDomainModel, self).__init__()
        self.embed_size = 128
        self.hidden_size = 256
        self.seq_length = configs.seq_len
        self.feature_size = configs.enc_in
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.number = configs.number
        
    
        # Linear layers for replacing MLP
        self.fc = nn.Sequential(
            nn.Linear(115*128, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 105)
        )
        
        # Embedding & Attention layers
        self.layer_norm = nn.LayerNorm(128)
        self.linear_128 = nn.Linear(128, 1)
        self.linear_seq = nn.Linear(self.seq_length, 1)
        self.linear_temporal = nn.Linear(115, 1)
        self.dct_layer = dct_channel_block(self.seq_length)
        self.attention = MultiHeadAttention(self.embed_size, configs.n_heads)

        # weights for real and imaginary parts
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

    def token_emb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        return x * self.embeddings

    def frequency_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.frequency_mlp(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    def frequency_sequence(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.frequency(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size -1, dim=2, norm="ortho")
        return x.permute(0, 2, 1, 3)

    def frequency(self, B, nd, dimension, x, r, i, rb, ib):
        real_part = F.relu(torch.einsum('bijd,dd->bijd', x.real, r) - torch.einsum('bijd,dd->bijd', x.imag, i) + rb)
        imag_part = F.relu(torch.einsum('bijd,dd->bijd', x.imag, r) + torch.einsum('bijd,dd->bijd', x.real, i) + ib)
        y = torch.stack([real_part, imag_part], dim=-1)
        return torch.view_as_complex(F.softshrink(y, lambd=0.01))

    def forward(self, x):
        B, T, N = x.shape
        x2 = self.token_emb(x)
        number = self.number
        X = x2[:, -1:, :, :]
        Y = x2[:, :N-1, :, :]
        x2 = self.frequency_sequence(Y, B, N-1, T)
        weights = torch.tensor([0.5**i for i in range(N-1)], dtype=torch.float32).to(x2.device)
        weights /= weights.sum()

        X_105 = (x2[:,:N-1, number:, :] * weights.view(1, N -1, 1, 1)).sum(dim=1, keepdim=True)
        updated_X = torch.cat((X[:, :, :number, :], X_105), dim=2)
        x2 = updated_X

        x2 = self.frequency_temporal(x2, B, N, T)
        x = x2[:, -1, :, :].squeeze(1)

        x = self.dct_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.linear_128(x).squeeze(-1)
        mul_out = output[:, number:]
        single_output = self.linear_seq(output)

        return single_output.unsqueeze(-1), mul_out.unsqueeze(-1)
