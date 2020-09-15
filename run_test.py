# coding: utf-8
import h5py as h5 
from model import * 
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from PIL import Image 


a = h5.File('f.h5','r')
a = a['array'][0][:,:,1]
model = Model()
c = model(torch.Tensor(resize(a,(512,512))).unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()
print(c.shape)
