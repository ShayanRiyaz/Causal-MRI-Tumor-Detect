import torch
import torch.nn as nn
from MyViT.ViT import MyViT
from CausalVAE.causal_vae import CausalVAE

class HybridModel(nn.Module):
    def __init__(self,vit_config,vae_input_dim,vae_z_dim = 50,vae_hidden_dim = 400,num_classes=2,batch_size=128):
        super(HybridModel,self).__init__()
        self.vit = MyViT(**vit_config)
        self.causal_vae = CausalVAE(input_dim=vae_input_dim,z_dim=vae_z_dim,hidden_dim = vae_hidden_dim)
        vit_feature_dim = vit_config.get('hidden_d',8)
        fusion_dim = vit_feature_dim + vae_z_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim,batch_size),
            nn.ReLU(),
            nn.Linear(batch_size,num_classes),# nn.Softmax(dim=-1) 
        )


    def forward(self,images, return_all=False):
        vit_features = self.vit.get_features(images)

        batch_size = images.size(0)
        images_flat = images.view(batch_size,-1)
        x_recon, z, z_mu, z_logvar = self.causal_vae(images_flat)

        fused = torch.cat([vit_features,z],dim=1)
        logits = self.fusion_layer(fused)

        if return_all:
            # Return classification logits and VAE outputs for computing reconstruction and KL losses.
            return logits, x_recon, z_mu, z_logvar
        else:
            return logits