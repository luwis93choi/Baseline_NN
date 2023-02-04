import h5py
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

class dataset_compiler():

    def __init__(self, dataset_save_path='',
                       train_sequeunce=['0000'],
                       valid_sequence=['0001'],
                       test_sequence=['0002'],
                       verbose='low'):

        self.dataset_save_path = dataset_save_path

        self.train_sequeunce = train_sequeunce
        self.valid_sequence = valid_sequence
        self.test_sequence = test_sequence

        self.verbose = verbose

        
        ### Dataset HDF Preparation by Dataset Group Type ###
        main_file = h5py.File(self.dataset_save_path, 'w')

        train_group = main_file.create_group('train_group')
        train_group.attrs['sequence'] = train_sequeunce
        train_group.attrs['type'] = 'train'
        #####################################################

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)