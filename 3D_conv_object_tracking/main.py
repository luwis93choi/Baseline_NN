import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import device
from torch.utils.data import DataLoader

import numpy as np

import h5py
from tqdm import tqdm
import datetime

from dataset_compiler import dataset_compiler
from dataset import object_tracking_dataset

from model import Object_Tracking_VIT
from model import Object_Tracking_UNet
from model import Object_Tracking_2DConvNet

from torch.utils.tensorboard import SummaryWriter

### Hyper Parameter ###
BATCH_SIZE = 32
EPOCH = 1000

LEARNING_RATE = 1e-5
LEARNING_RATE_SCHEDULING_PERIOD = 20

OBJECTNESS_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 1.0

THREAD_NUM = 4

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNEL = 3

LABEL_MASK_WIDTH = 64
LABEL_MASK_HEIGHT = 64

### Save Directory Setup ###
start_time = str(datetime.datetime.now())
start_time = start_time.replace(':', '_')
start_time = start_time.replace('-', '_')
start_time = start_time.replace('_', '_')
start_time = start_time.replace('.', '_')
start_time = start_time.replace(' ', '_')

### Processor Setup ###
processor = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(processor)
print(processor)
'''
dataset_compiler(dataset_save_path='./dataset.hdf5',
                 image_dataset_path='/home/luwis/ICSL_Project/Datasets/BDD_100K/bdd100k_images_100k/bdd100k/images/100k',
                 groundtruth_dataset_path='/home/luwis/ICSL_Project/Datasets/BDD_100K/bdd100k_labels_release/bdd100k/labels',
                 original_input_image_height=720,
                 original_input_image_width=1280,
                 output_grid_height=LABEL_MASK_HEIGHT,
                 output_grid_width=LABEL_MASK_WIDTH,
                 detection_labels=['person', 'car', 'truck', 'motor', 'bike', 'bus'],
                 detection_type='center_point',
                 verbose='low')
'''
train_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', target_input_image_height=IMAGE_HEIGHT, target_input_image_width=IMAGE_WIDTH, data_augmentation_prob=0, mode='train', verbose='low')
train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=THREAD_NUM)

valid_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', target_input_image_height=IMAGE_HEIGHT, target_input_image_width=IMAGE_WIDTH, data_augmentation_prob=0, mode='valid', verbose='low')
valid_dataloader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=THREAD_NUM)

tracking_net = Object_Tracking_2DConvNet(bias=True, verbose='low')
tracking_net.to(processor)

optimizer = optim.Adam(tracking_net.parameters(), lr=LEARNING_RATE)

objectness_loss = nn.BCEWithLogitsLoss()

### Summary Writer Setup ###
training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/3D_conv_object_tracking_training')
valid_writer = SummaryWriter(log_dir='./runs/' + start_time + '/3D_conv_object_tracking_validation')

for epoch in range(EPOCH):

    train_total_loss_list = []

    valid_total_loss_list = []

    print('[EPOCH : {}]'.format(epoch))

    ### Training Loop ###
    tracking_net.train()

    iteration_length = len(train_dataloader)

    for batch_idx, (input_img, groundtruth_objectness_mask_img) in enumerate(tqdm(train_dataloader)):
    
        input_img = input_img.to(processor).float()
        #print(input_img.size())

        groundtruth_objectness_mask_img = groundtruth_objectness_mask_img.to(processor).float()
        #print(groundtruth_objectness_mask_img.size())
      
        optimizer.zero_grad()

        net_output = tracking_net(input_img)

        obj_loss = objectness_loss(net_output, groundtruth_objectness_mask_img)

        loss_val = OBJECTNESS_LOSS_WEIGHT * obj_loss
        loss_val.backward()
        
        optimizer.step()

        train_total_loss_list.append(loss_val.item())
    
    training_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(train_total_loss_list), epoch)
 
    if (epoch % (LEARNING_RATE_SCHEDULING_PERIOD//2)) == 0:
        training_imgs = input_img.clone().detach().cpu().numpy()
        training_GT = groundtruth_objectness_mask_img.clone().detach().cpu().numpy()
        training_outputs = net_output.clone().detach().cpu().numpy()
        training_writer.add_images('Sample Inputs - Training [{}]'.format(start_time), training_imgs, epoch)
        training_writer.add_images('Sample GT - Training [{}]'.format(start_time), training_GT, epoch)
        training_writer.add_images('Sample Outputs - Training [{}]'.format(start_time), training_outputs, epoch)

    ### Validation Loop ###
    tracking_net.eval()

    with torch.no_grad():
        
        for batch_idx, (input_img, groundtruth_objectness_mask_img) in enumerate(tqdm(valid_dataloader)):
        
            input_img = input_img.to(processor).float()

            groundtruth_objectness_mask_img = groundtruth_objectness_mask_img.to(processor).float()
            
            net_output = tracking_net(input_img)

            obj_loss = objectness_loss(net_output, groundtruth_objectness_mask_img)
            loss_val = OBJECTNESS_LOSS_WEIGHT * obj_loss

            valid_total_loss_list.append(loss_val.item())

        valid_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(valid_total_loss_list), epoch)

        if (epoch % (LEARNING_RATE_SCHEDULING_PERIOD//2)) == 0:
            valid_imgs = input_img.clone().detach().cpu().numpy()
            valid_GT = groundtruth_objectness_mask_img.clone().detach().cpu().numpy()
            valid_outputs = net_output.clone().detach().cpu().numpy()
            valid_writer.add_images('Sample Inputs - Validation [{}]'.format(start_time), valid_imgs, epoch)
            valid_writer.add_images('Sample GT - Validation [{}]'.format(start_time), valid_GT, epoch)
            valid_writer.add_images('Sample Outputs - Validation [{}]'.format(start_time), valid_outputs, epoch)

    if np.average(valid_total_loss_list) <= np.average(train_total_loss_list):

        torch.save({'epoech': epoch,
                    'model_state_dict': tracking_net.state_dict(),
                    'training_loss': np.average(train_total_loss_list),
                    'valid_loss': np.average(valid_total_loss_list)},
                    './runs/' + start_time + '/3D_conv_object_tracking_state_dict.pth')
        
        torch.save(tracking_net, './runs/' + start_time + '/3D_conv_object_tracking.pth')

        '''
        dummy_input = torch.randn(1, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH).to(processor).float()
        dummy_output = tracking_net(dummy_input)

        torch.onnx.export(
            tracking_net,
            dummy_input,
            './runs/' + start_time + '/3D_conv_object_tracking_state_dict.onnx',
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            opset_version=13
        )
        '''