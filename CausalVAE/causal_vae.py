import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,input_dim,z_dim,hidden_dim):
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim,z_dim)
        self.fc_logvar = nn.Linear(hidden_dim,z_dim)


    def forward(self,x):
        h = F.softplus(self.fc1(x))
        z_mu = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        return z_mu, z_logvar
    
class Decoder(nn.Module):
    def __init__(self,z_dim,output_dim,hidden_dim):
        super(Decoder,self).__init__()
        self.fc1 = nn.Linear(z_dim,hidden_dim)
        self.fc_out = nn.Linear(hidden_dim,output_dim)

    def forward(self,z):
        h = F.softplus(self.fc1(z))
        x_recon = torch.sigmoid(self.fc_out(h))
        return x_recon


class CausalVAE(nn.Module):
    def __init__(self,input_dim,z_dim=50,hidden_dim=400):
        super(CausalVAE,self).__init__()
        self.encoder = Encoder(input_dim,z_dim,hidden_dim)
        self.decoder = Decoder(z_dim,input_dim,hidden_dim)

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        z_mu,z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu,z_logvar)
        x_recon = self.decoder(z)
        return x_recon,z,z_mu,z_logvar
