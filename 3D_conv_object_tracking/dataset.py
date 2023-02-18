import h5py

import cv2 as cv

import numpy as np

import torch
import torch.utils.data

class object_tracking_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_save_path='',
                       target_input_image_height=98,
                       target_input_image_width=320,
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

        img_path_prev = str(img_path_list[0], 'utf-8')
        img_path_current = str(img_path_list[1], 'utf-8')

        self.local_print(img_path_prev, level='high')
        self.local_print(img_path_current, level='high')

        prob = np.random.rand(1)[0]

        input_img_current = cv.imread(img_path_current, cv.IMREAD_COLOR)
        input_img_current = cv.resize(input_img_current, (self.target_input_image_width, self.target_input_image_height), interpolation=cv.INTER_AREA)
        
        if self.data_augmentation_prob >= prob: input_img_current = cv.flip(input_img_current, 1)
        
        input_img_current = np.transpose(input_img_current, (2, 0, 1))
        input_img_current = input_img_current / 255
        self.local_print('input_img_current : {}'.format(input_img_current.shape), level='high')
        
        ### Groundtruth Label Preparation ##########################
        groundtruth_label_matrix = self.groundtruth_group[idx][()]

        groundtruth_objectness_mask_img = groundtruth_label_matrix[:, :, 0]
        
        if self.data_augmentation_prob >= prob: groundtruth_objectness_mask_img = cv.flip(groundtruth_objectness_mask_img, 1)
        ############################################################

        return input_img_current, groundtruth_objectness_mask_img

        '''
        input_img_prev = cv.imread(img_path_prev, cv.IMREAD_COLOR)
        input_img_prev = cv.resize(input_img_prev, (self.target_input_image_width, self.target_input_image_height))
        input_img_prev = np.transpose(input_img_prev, (2, 0, 1))
        input_img_prev = np.expand_dims(input_img_prev, axis=1)
        input_img_prev = input_img_prev / 255
        self.local_print('input_img_prev : {}'.format(input_img_prev.shape), level='high')

        input_img_current = cv.imread(img_path_current, cv.IMREAD_COLOR)
        input_img_current = cv.resize(input_img_current, (self.target_input_image_width, self.target_input_image_height))
        input_img_current = np.transpose(input_img_current, (2, 0, 1))
        input_img_current = np.expand_dims(input_img_current, axis=1)
        input_img_current = input_img_current / 255
        self.local_print('input_img_current : {}'.format(input_img_current.shape), level='high')
        
        stacked_input_img = np.concatenate((input_img_prev, input_img_current), axis=1)
        self.local_print('stacked_input_img : {}'.format(stacked_input_img.shape), level='high')

        ### Groundtruth Label Preparation ##########################
        groundtruth_label_matrix = self.groundtruth_group[idx][()]

        groundtruth_label_matrix = np.transpose(groundtruth_label_matrix, (2, 0, 1))
        ############################################################
        
        return stacked_input_img, groundtruth_label_matrix
        '''
        
    def __len__(self):

        return self.len
    
    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)