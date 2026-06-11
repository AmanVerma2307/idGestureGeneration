import os
import torch
import numpy as np

class scut(torch.utils.data.Dataset):

    """
    Dataset class for SCUT-DHGA
    """

    def __init__(self,
                 dataDir,
                 mode,
                 splitSize):
        
        self.dataDir = './datasets/'
