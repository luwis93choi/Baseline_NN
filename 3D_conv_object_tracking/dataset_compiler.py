import h5py
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

class dataset_compiler():

    def __init__(self, dataset_save_path='',
                       image_dataset_path='',
                       groundtruth_dataset_path='',
                       original_input_image_height=192,
                       original_input_image_width=640,
                       output_grid_height=12,
                       output_grid_width=40,
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
                    
                    data_length = len(sorted(os.listdir(img_base_path)))
                    
                    img_data_name_prev = sorted(os.listdir(img_base_path))[1:]
                    img_data_name_current = sorted(os.listdir(img_base_path))[:data_length-1]

                    self.local_print('{}'.format(img_data_name_prev), level='high')
                    self.local_print('{}'.format(img_data_name_current), level='high')

                    ### Groundtruth Label Accumulation ###
                    with open(self.groundtruth_dataset_path + '/' + sequence + '.txt', 'r') as groundtruth_file:
                        groundtruth_lines = groundtruth_file.readlines()

                    self.local_print('{}'.format(groundtruth_lines), level='high')

                    groundtruth_matrix = [x.strip().split() for x in groundtruth_lines]
                    groundtruth_matrix = np.array(groundtruth_matrix)
                    self.local_print('{}'.format(groundtruth_matrix), level='high')

                    ### Groundtruth Data Accumulation ###
                    self.local_print('[Dataset Compiler] Type : {} | Groundtruth Sequence : {}'.format(group.name, sequence), level='low')
                    
                    ### Iterate through the dataset ###
                    for idx, (prev_img, current_img) in enumerate(zip(tqdm(img_data_name_prev), tqdm(img_data_name_current))):
                        
                        if idx < self.skip_count: pass
                        else:

                            ### YOLO Groundtruth Label Matrix Generation + Write ###
                            class_label_num = 8     # DontCare class needs to be ignored
                            bbox_label_num = 2 * 2
                            occulded_label_num = 1
                            
                            label_matrix_channel = occulded_label_num + class_label_num

                            groundtruth_label_matrix = np.zeros((output_grid_height, output_grid_width, label_matrix_channel), dtype=float)
                            
                            groundtruth_idx = np.where(groundtruth_matrix[:, 0] == str(idx))

                            current_labels = groundtruth_matrix[groundtruth_idx, :]
                            print(idx)
                            print(current_labels.shape)

                            for label in current_labels[0, :, :]:

                                if label[2] == 'DontCare': pass
                                else:
                                    # Target Class Assignment
                                    if label[2] == 'Car':               class_type = 0
                                    elif label[2] == 'Van':             class_type = 1
                                    elif label[2] == 'Truck':           class_type = 2
                                    elif label[2] == 'Pedestrian':      class_type = 3
                                    elif label[2] == 'Person_sitting':  class_type = 4
                                    elif label[2] == 'Cyclist':         class_type = 5
                                    elif label[2] == 'Tram':            class_type = 6
                                    elif label[2] == 'Misc':            class_type = 7

                                    # Objectness Assignment based on Occulusion Label
                                    if label[4] == '0':       objectness = 1      # Fully Visible
                                    elif label[4] == '1':     objectness = 0.7    # Partially Occuluded
                                    elif label[4] == '2':     objectness = 0.3    # Largely Occuluded
                                    elif label[4] == '3':     objectness = 0      # Unknown

                                    print(objectness)

                            print('----------------')

                            ### Input Image Stack Write ###
                            img_path_list = [img_base_path + '/' + prev_img, 
                                             img_base_path + '/' + current_img]

                            img_path_group.create_dataset(name=str(data_idx).zfill(10),
                                                          data=img_path_list,
                                                          compression='gzip', compression_opts=9)

                            data_idx += 1

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)