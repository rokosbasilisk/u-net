import h5py as h5 
import torch
import numpy as np
class RoadMap(object):
    def __init__(self,array_path):
        self.array_path = array_path
        self.data_array = np.array(h5.File(self.array_path,'r')['array'])
if __name__ == '__main__':
    d = RoadMap('./f.h5')
    print(d.data_array.shape)
