import h5py
import os
import json

import cv2 as cv
import numpy as np

from tqdm import tqdm

import time

class dataset_compiler():

    def __init__(self, dataset_save_path='',
                       image_dataset_path='',
                       groundtruth_dataset_path='',
                       original_input_image_height=720,
                       original_input_image_width=1280,
                       output_grid_height=256,
                       output_grid_width=256,
                       detection_labels=['person', 'pedestrian', 'rider', 'car', 'truck', 'train', 'motorcycle', 'bicycle'],
                       detection_type='center_point',
                       verbose='low'):
        
        self.dataset_save_path = dataset_save_path

        self.image_dataset_path = image_dataset_path
        self.groundtruth_dataset_path = groundtruth_dataset_path

        self.original_input_image_height = original_input_image_height
        self.original_input_image_width = original_input_image_width

        self.output_grid_height = output_grid_height
        self.output_grid_width = output_grid_width

        self.detection_labels = detection_labels

        self.detection_type = detection_type

        self.verbose = verbose

        ### Dataset HDF Preparation by Dataset Group Type ###
        main_file = h5py.File(self.dataset_save_path, 'w')

        train_group = main_file.create_group('train_group')
        train_group.attrs['type'] = 'train'
        train_group.attrs['img_dir'] = self.image_dataset_path + '/train'
        train_group.attrs['label_file'] = self.groundtruth_dataset_path + '/' + 'bdd100k_labels_images_train.json'

        valid_group = main_file.create_group('valid_group')
        valid_group.attrs['type'] = 'valid'
        valid_group.attrs['img_dir'] = self.image_dataset_path + '/val'
        valid_group.attrs['label_file'] = self.groundtruth_dataset_path + '/' + 'bdd100k_labels_images_val.json'
        
        test_group = main_file.create_group('test_group')
        test_group.attrs['type'] = 'test'
        test_group.attrs['img_dir'] = self.image_dataset_path + '/test'
        #####################################################

        for group in [train_group, valid_group]:

            ### Dataset Accumulation ###
            self.local_print('[Dataset Compiler] Type : {}'.format(group.name), level='low')
            
            label_json_path = group.attrs['label_file']

            label_json_file = open(label_json_path)

            labels = json.load(label_json_file)

            img_path_group = main_file.create_group(group.name + '/img_path')
            groundtruth_group = main_file.create_group(group.name + '/groundtruth')

            ### Iterate through the dataset ###
            for idx, (label) in enumerate(tqdm(labels)):
                
                img_path = group.attrs['img_dir'] + '/' + label['name']
                img_labels = label['labels']

                groundtruth_label_img = np.zeros((output_grid_height, output_grid_width, 1), dtype=np.uint8)
                groundtruth_class_label_img = np.zeros((output_grid_height, output_grid_width, 1), dtype=np.uint8)

                if self.verbose == 'high':
                    img = cv.imread(img_path)
                    cv.imwrite('./test_original.png', img)

                    output_img = cv.resize(img, (self.output_grid_width, self.output_grid_height))
                    output_gt_circle_img = cv.resize(img, (self.output_grid_width, self.output_grid_height))
                    cv.imwrite('./test_output.png', output_img)

                for img_label in img_labels:

                    if img_label['category'] in self.detection_labels:

                        if img_label['attributes']['occluded'] == False:

                            bbox_label = img_label['box2d']

                            GT_top_left_x = bbox_label['x1']
                            GT_top_left_y = bbox_label['y1']
                            GT_bottom_right_x = bbox_label['x2']
                            GT_bottom_right_y = bbox_label['y2']

                            GT_center_x = (GT_top_left_x + GT_bottom_right_x) / 2
                            GT_center_y = (GT_top_left_y + GT_bottom_right_y) / 2

                            GT_center_x_ratio = GT_center_x / self.original_input_image_width
                            GT_center_y_ratio = GT_center_y / self.original_input_image_height

                            GT_top_left_x_ratio = GT_top_left_x / self.original_input_image_width
                            GT_top_left_y_ratio = GT_top_left_y / self.original_input_image_height
                            GT_bottom_right_x_ratio = GT_bottom_right_x / self.original_input_image_width
                            GT_bottom_right_y_ratio = GT_bottom_right_y / self.original_input_image_height

                            target_center_x = int(GT_center_x_ratio * self.output_grid_width)
                            target_center_y = int(GT_center_y_ratio * self.output_grid_height)

                            target_top_left_x = int(GT_top_left_x_ratio * self.output_grid_width)
                            target_top_left_y = int(GT_top_left_y_ratio * self.output_grid_height)
                            target_bottom_right_x = int(GT_bottom_right_x_ratio * self.output_grid_width)
                            target_bottom_right_y = int(GT_bottom_right_y_ratio * self.output_grid_height)

                            if detection_type == 'center_point':
                                groundtruth_label_img[target_center_y, target_center_x, 0] = 1

                                # Class 0 : No Target
                                groundtruth_class_label_img[target_center_y, target_center_x, 0] = self.detection_labels.index(img_label['category']) + 1
    
                            if detection_type == 'box_area':
                                groundtruth_label_img[target_top_left_y : target_bottom_right_y + 1, target_top_left_x : target_bottom_right_x + 1, 0] = 1
                                
                                # Class 0 : No Target
                                groundtruth_class_label_img[target_top_left_y : target_bottom_right_y + 1, target_top_left_x : target_bottom_right_x + 1, 0] = self.detection_labels.index(img_label['category']) + 1

                            if self.verbose == 'high':

                                gt_img = cv.rectangle(img, (int(GT_top_left_x), int(GT_top_left_y)), (int(GT_bottom_right_x), int(GT_bottom_right_y)), (255, 255, 255), 2)
                                cv.circle(output_gt_circle_img, center=(target_center_x, target_center_y), radius=5, color=(255, 255, 255), thickness=2)

                                recovered_target_center_x = int(self.original_input_image_width * (target_center_x / self.output_grid_width))
                                recovered_target_center_y = int(self.original_input_image_height * (target_center_y / self.output_grid_height))
                                cv.circle(gt_img, center=(recovered_target_center_x, recovered_target_center_y), radius=5, color=(255, 255, 255), thickness=2)

                if self.verbose == 'high':

                    cv.imwrite('./test_bbox.png', gt_img)
                    cv.imwrite('./test_center_circle.png', output_gt_circle_img)

                    groundtruth_label_img = 255 * np.repeat(groundtruth_label_img, repeats=3, axis=2)
                    added_img = cv.addWeighted(output_img, 0.6, groundtruth_label_img, 0.4, 0)
                    cv.imwrite('./test_center.png', added_img)

                    time.sleep(1)
                    
                ### Input Image Data Write ###
                img_path_group.create_dataset(name=str(idx).zfill(10),
                                              data=[img_path],
                                              compression='gzip', compression_opts=9)

                groundtruth_group.create_dataset(name=str(idx).zfill(10),
                                                 data=np.concatenate((groundtruth_label_img, groundtruth_class_label_img), axis=2),
                                                 compression='gzip', compression_opts=9)

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)