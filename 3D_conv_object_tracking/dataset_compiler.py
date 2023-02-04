import h5py
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

class dataset_compiler():

    def __init__(self, dataset_save_path='',
                       image_dataset_path='',
                       groundtruth_dataset_path='',
                       train_sequence=['0000'],
                       valid_sequence=['0001'],
                       test_sequence=['0002'],
                       skip_count=1,
                       verbose='low'):

        self.dataset_save_path = dataset_save_path

        self.image_dataset_path = image_dataset_path
        self.groundtruth_dataset_path = groundtruth_dataset_path

        self.train_sequence = train_sequence
        self.valid_sequence = valid_sequence
        self.test_sequence = test_sequence

        self.verbose = verbose

        self.skip_count = skip_count

        ### Dataset HDF Preparation by Dataset Group Type ###
        main_file = h5py.File(self.dataset_save_path, 'w')

        train_group = main_file.create_group('train_group')
        train_group.attrs['sequence'] = self.train_sequence
        train_group.attrs['type'] = 'train'

        valid_group = main_file.create_group('valid_group')
        valid_group.attrs['sequence'] = self.valid_sequence
        valid_group.attrs['type'] = 'valid'
        
        test_group = main_file.create_group('test_group')
        test_group.attrs['sequence'] = self.test_sequence
        test_group.attrs['type'] = 'test'
        #####################################################

        for group in [train_group, valid_group, test_group]:

            img_path_group = main_file.create_group(group.name + '/img_path')
            groundtruth_group = main_file.create_group(group.name + '/groundtruth')

            data_idx = 0

            if len(group.attrs['sequence']) > 0:

                for sequence_idx, (sequence) in enumerate(group.attrs['sequence']):

                    ### Image Path Accumulation ###
                    self.local_print('[Dataset Compiler] Type : {} | Image Data Sequence : {}'.format(group.name, sequence), level='low')
                    img_base_path = self.image_dataset_path + '/' + sequence
                    img_data_name_prev = sorted(os.listdir(img_base_path))
                    img_data_name_current = sorted(os.listdir(img_base_path))

                    self.local_print('{}'.format(img_data_name_prev), level='high')
                    self.local_print('{}'.format(img_data_name_current), level='high')

                    ### Groundtruth Data Accumulation ###
                    self.local_print('[Dataset Compiler] Type : {} | Groundtruth Sequence : {}'.format(group.name, sequence), level='low')
                    
                    with open(self.groundtruth_dataset_path + '/' + sequence + '.txt', 'r') as groundtruth_file:
                        groundtruth_lines = groundtruth_file.readlines

                    self.local_print('{}'.format(groundtruth_lines), level='high')

                    ### Iterate through the dataset ###
                    for idx, (prev_img, current_img) in enumerate(zip(tqdm(img_data_name_prev), tqdm(img_data_name_current))):
                        
                        if idx < self.skip_count: pass
                        else:

                            ### Add Groundtruth Label Processing #######################


                            ############################################################

                            img_path_list = [img_base_path + '/' + img_data_name_prev[idx-skip_count], 
                                             img_base_path + '/' + img_data_name_prev[idx]]

                            img_path_group.create_dataset(name=str(data_idx).zfill(10),
                                                          data=img_path_list,
                                                          compression='gzip', compression_opts=9)

                            data_idx += 1

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)