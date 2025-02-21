import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, n_feat, d_model):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_feat)
        self.decoder = nn.Linear(n_feat, d_model)
        self.relu = nn.ReLU() # this is not best SAE activation

    def encode(self, x_in):
        x = x_in - self.decoder.bias
        f = self.relu(self.encoder(x))
        return f
    
    def forward(self, x_in, compute_loss=False, compute_ap=False):
        f = self.encode(x_in)
        x = self.decoder(f)
        if compute_loss:
            recon_loss = F.mse_loss(x, x_in)
            reg_loss = f.abs().sum(dim=-1).mean()
        else:
            recon_loss = None
            reg_loss = None
        if compute_ap:
            ap = (f > 0).sum(dim=-1).mean()
            return x, recon_loss, reg_loss, ap
        return x, recon_loss, reg_loss
    
    def normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, p=2, dim=1)
