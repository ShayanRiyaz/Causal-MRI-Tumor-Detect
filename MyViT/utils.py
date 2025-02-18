import torch 
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image
import cv2 as cv


# Reshape (N,1,C,R) into (N,P^2,HWC/P^2)
# Where P = Desired Patch Size
# Convert Image into patches for transformer input
def patchify(images,n_patches):
    n,c,h,w = images.shape
    
    assert h == w, " Patchify method can be implemented fpr square images only"

    patches = torch.zeros(n,n_patches**2,h*w*c//n_patches**2)
    patch_size = h // n_patches
    
    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                patches[idx,i*n_patches+j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length,d):
    result = torch.ones(sequence_length,d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i/(10000**(j/d))) 
            else:
                result[i][j] = np.cos(i/(10000**((j-1)/d)))
    return result


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None,image_shape=128,random_modification =False):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_shape = image_shape
        self.random_modification = random_modification

        tumor_dir = os.path.join(root_dir, 'tumor')
        non_tumor_dir = os.path.join(root_dir, 'notumor')

        for file in os.listdir(tumor_dir):
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                self.image_paths.append(os.path.join(tumor_dir, file))
                self.labels.append(1)

        for file in os.listdir(non_tumor_dir):
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                self.image_paths.append(os.path.join(non_tumor_dir, file))
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # image = Image.open(img_path).convert('L')
        image = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
        image = cv.resize(image,(self.image_shape , self.image_shape ))

        if self.random_modification:
            image = cv.equalizeHist(image)
            height, width = image.shape[:2]
            center = (width/2, height/2)
            rotate_matrix = cv.getRotationMatrix2D(center=center, angle=45, scale=1)
            
            if np.round(np.random.rand()) == 1:
                if np.round(np.random.rand()) == 1:
                    image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
                else:
                    image = cv.flip(image, 0) 
         
        if self.transform:
            image = self.transform(image)
        return image, label
    

class MRISegDataset(Dataset):
    def __init__(self, root_dir, transform=None,image_shape=28,random_modification =False):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_shape = image_shape
        self.random_modification = random_modification

        for file in sorted(os.listdir(root_dir)):
            if file.lower().endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, file))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # image = Image.open(img_path).convert('L')
        image = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
        image = cv.resize(image,(self.image_shape , self.image_shape ))

        if self.random_modification == 1:
            image = cv.equalizeHist(image)
            height, width = image.shape[:2]
            center = (width/2, height/2)
            rotate_matrix = cv.getRotationMatrix2D(center=center, angle=45, scale=1)
            
            if np.round(np.random.rand()) == 1:
                if np.round(np.random.rand()) == 1:
                    image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
                else:
                    image = cv.flip(image, 0) 
        
        if self.transform:
            image = self.transform(image)
        return image#, label