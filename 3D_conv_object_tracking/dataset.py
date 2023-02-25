import h5py

import cv2 as cv

import numpy as np

import torch
import torch.utils.data

class object_tracking_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_save_path='',
                       target_input_image_height=256,
                       target_input_image_width=256,
                       data_augmentation_prob=0.1,
                       mode='train',
                       verbose='low'):
        
        self.dataset_save_path = dataset_save_path

        self.dataset_file = h5py.File(dataset_save_path)

        self.mode = mode

        self.data_augmentation_prob = data_augmentation_prob

        self.verbose = verbose

        self.target_input_image_height = target_input_image_height
        self.target_input_image_width = target_input_image_width

        if self.mode == 'train':
            self.img_path_group = self.dataset_file['/train_group/img_path']
            self.groundtruth_group = self.dataset_file['/train_group/groundtruth']

        elif self.mode == 'valid':
            self.img_path_group = self.dataset_file['/valid_group/img_path']
            self.groundtruth_group = self.dataset_file['/valid_group/groundtruth']
            
        elif self.mode == 'test':
            self.img_path_group = self.dataset_file['/test_group/img_path']
            self.groundtruth_group = self.dataset_file['/test_group/groundtruth']

        self.len = self.img_path_group.__len__()

        print('[Dataset Status]')
        print('dataset_path : {}'.format(self.dataset_save_path))
        print('dataset_file : {}'.format(self.dataset_file))
        print('dataset_mode : {}'.format(self.mode))
        print('dataset_length : {}'.format(self.len), end='\n\n')

    def __getitem__(self, index):

        ### Index Prepration ###
        idx = str(index).zfill(10)

        ### Input Image Preparation ###
        img_path_list = self.img_path_group[idx][()]

        img_path = str(img_path_list[0], 'utf-8')

        self.local_print(img_path, level='high')

        input_img = cv.imread(img_path, cv.IMREAD_COLOR)
        self.local_print('Original Input Image Shape : {}'.format(input_img.shape), level='high')
        
        input_img = cv.resize(input_img, (self.target_input_image_width, self.target_input_image_height), interpolation=cv.INTER_AREA)
        
        # 3 Channel Input Image (Channel Last)
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = input_img / 255
        self.local_print('input_img : {}'.format(input_img.shape), level='high')
        
        ### Groundtruth Label Preparation ##########################
        groundtruth_label_matrix = self.groundtruth_group[idx][()]

        groundtruth_objectness_mask_img = groundtruth_label_matrix[:, :, 0]
        self.local_print('groundtruth_objectness_mask_img : {}'.format(groundtruth_objectness_mask_img.shape), level='high')

        groundtruth_objectness_mask_img = np.expand_dims(groundtruth_objectness_mask_img, axis=0)
        self.local_print('groundtruth_objectness_mask_img : {}'.format(groundtruth_objectness_mask_img.shape), level='high')
        ############################################################

        return input_img, groundtruth_objectness_mask_img
        
    def __len__(self):

        return self.len
    
    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)