import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from PIL import Image

from tqdm import tqdm, trange
import torchvision.transforms as transforms
import numpy as np

from MyViT.utils import get_positional_embeddings,patchify,MRIDataset


    
class MyViT(nn.Module):
    def __init__(self,chw,n_patches = 8, n_blocks=2,hidden_d=8,n_heads=2,out_d=10):
        super(MyViT,self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
    
        assert chw[1] % n_patches == 0, "Input shape not divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not divisible by number of patches"
        self.patch_size = (chw[1]/n_patches,chw[2]/n_patches)
        # 1) Linear Mapper
        self.input_d = int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Class Token
        self.class_token = nn.Parameter(torch.rand(1,self.hidden_d))

        # 3) Positional Embedding
        # self.pos_embed = nn.Parameter(
        #                 torch.tensor(
        #                 get_positional_embeddings(self.n_patches**2+1,self.hidden_d)
        #                 )
        #             )
        # self.pos_embed_requires_grad = False
        self.register_buffer('positional_embeddings',get_positional_embeddings(n_patches**2+1,hidden_d),persistent=False)
   
        # 4 ) Transformer Encoder Blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d,n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(nn.Linear(self.hidden_d,out_d),
                                 nn.Softmax(dim = -1)
                                 )
    def forward(self,images):
        n,c,h,w = images.shape
        patches = patchify(images,self.n_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)
        
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n,1,-1), tokens), dim = 1)
        out = tokens+self.positional_embeddings.repeat(n,1,1)

        for block in self.blocks:
            out = block(out)

        out = out[:,0]
        return self.mlp(out)
    

class MyMSA(nn.Module):
    def __init__(self,d,n_heads =2):
        super(MyMSA,self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads}"
        
        d_head = int(d/n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head,d_head)for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head,d_head)for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head,d_head)for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,sequences):
        # Sequences has shape (N,seq_length,token_dim)
        # We go into shape (N, seq_length, n_heads,token_dim/n_heads)
        # And come back to (N, seq_length, item_dim) (through concatenation)

        result = []
        for sequence in sequences:
            sequence_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:,head*self.d_head:(head+1)*self.d_head]
                q,k,v = q_mapping(seq),k_mapping(seq),v_mapping(seq)

                attention = self.softmax(q @ k.T/(self.d_head)**0.5)
                sequence_result.append(attention @ v)
            result.append(torch.hstack(sequence_result))
        return torch.cat([torch.unsqueeze(r,dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self,hidden_d,n_heads,mlp_ratio = 4):
        super(MyViTBlock,self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d,n_heads)


        # Encoder Block
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d,mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d,hidden_d)
        )

    def forward(self,x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out