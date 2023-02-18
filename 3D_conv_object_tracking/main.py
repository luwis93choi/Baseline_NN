import torch
import torch.nn as nn
import torch.optim as optim
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

from torch.utils.tensorboard import SummaryWriter

### Hyper Parameter ###
BATCH_SIZE = 32
EPOCH = 1000

LEARNING_RATE = 5e-4

OBJECTNESS_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 1.0

THREAD_NUM = 2

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3

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

dataset_compiler(dataset_save_path='./dataset.hdf5',
                 current_image_dataset_path='/home/luwis/ICSL_Project/KITTI_dataset/Object_Video_Tracking/data_tracking_image_2/training/image_02',
                 prev_image_dataset_path='/home/luwis/ICSL_Project/KITTI_dataset/Object_Video_Tracking/data_tracking_image_2/training/image_02_prev',
                 groundtruth_dataset_path='/home/luwis/ICSL_Project/KITTI_dataset/Object_Video_Tracking/data_tracking_label_2/training/label_02',
                 original_input_image_height=375,
                 original_input_image_width=1242,
                 target_input_image_height=IMAGE_HEIGHT,
                 target_input_image_width=IMAGE_WIDTH,
                 output_grid_height=12,
                 output_grid_width=40,
                 train_sequence=['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0014', '0015', '0016', '0017', '0018', '0019'],
                 valid_sequence=['0011', '0012', '0013'],
                 test_sequence=['0020'],
                 skip_count=1,
                 verbose='low')

train_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', target_input_image_height=IMAGE_HEIGHT, target_input_image_width=IMAGE_WIDTH, data_augmentation_prob=0.45, mode='train', verbose='low')
train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=THREAD_NUM)

valid_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', target_input_image_height=IMAGE_HEIGHT, target_input_image_width=IMAGE_WIDTH, data_augmentation_prob=0, mode='valid', verbose='low')
valid_dataloader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=THREAD_NUM)

test_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', target_input_image_height=IMAGE_HEIGHT, target_input_image_width=IMAGE_WIDTH, data_augmentation_prob=0, mode='test', verbose='low')
test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=THREAD_NUM)

#tracking_net = Object_Tracking_UNet(bias=True, dropout_prob=0.1,verbose='low')
tracking_net = Object_Tracking_VIT(input_width=IMAGE_WIDTH,
                                    input_height=IMAGE_HEIGHT,
                                    input_channel=IMAGE_CHANNEL,
                                    reduce_channel=True,
                                    input_batchsize=BATCH_SIZE,
                                    patch_width=16,
                                    patch_height=16,
                                    transformer_embedding_bias=True,
                                    transformer_nhead=8,
                                    transforemr_internal_feedforward_embedding=1024,
                                    transformer_dropout=0.1,
                                    transformer_activation='gelu',
                                    transformer_encoder_layer_num=8,
                                    classification_layer_bias=True,
                                    device=processor,
                                    verbose='low')
tracking_net.to(processor)

optimizer = optim.Adam(tracking_net.parameters(), lr=LEARNING_RATE)

#objectness_loss = nn.BCEWithLogitsLoss()
objectness_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()

### Summary Writer Setup ###
training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/3D_conv_object_tracking_training')
valid_writer = SummaryWriter(log_dir='./runs/' + start_time + '/3D_conv_object_tracking_validation')

for epoch in range(EPOCH):

    train_total_loss_list = []
    train_objectness_loss_list = []
    train_classification_loss_list = []

    valid_total_loss_list = []
    valid_objectness_loss_list = []
    valid_classification_loss_list = []

    print('[EPOCH : {}]'.format(epoch))

    ### Training Loop ###
    tracking_net.train()

    for batch_idx, (input_img, groundtruth_objectness_mask_img) in enumerate(tqdm(train_dataloader)):
    
        #print(input_img.size())
        #print(groundtruth_objectness_mask_img.size())
        
        input_img = input_img.to(processor).float()
        #print(input_img.size())

        groundtruth_objectness_mask_img = groundtruth_objectness_mask_img.to(processor).float()
        #print(groundtruth_objectness_mask_img.size())

        #class_label_matrix = groundtruth_label_matrix[:, 1, :, :].to(processor).long()

        optimizer.zero_grad()

        net_output = tracking_net(input_img)

        obj_loss = objectness_loss(net_output, groundtruth_objectness_mask_img)

        loss_val = OBJECTNESS_LOSS_WEIGHT * obj_loss
        loss_val.backward()
        
        optimizer.step()

        train_total_loss_list.append(loss_val.item())
        #train_objectness_loss_list.append(obj_loss.item())
        #train_CrossEntropy_loss_list.append(class_loss.item())
        
    training_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(train_total_loss_list), epoch)
    #training_writer.add_scalar('Average MSE Loss per Epoch - {}'.format(start_time), np.average(train_objectness_loss_list), epoch)
    #training_writer.add_scalar('Average CrossEntropy Loss per Epoch - {}'.format(start_time), np.average(train_CrossEntropy_loss_list), epoch)

    ### Validation Loop ###
    tracking_net.eval()

    with torch.no_grad():
        
        for batch_idx, (input_img, groundtruth_objectness_mask_img) in enumerate(tqdm(valid_dataloader)):
        
            input_img = input_img.to(processor).float()

            groundtruth_objectness_mask_img = groundtruth_objectness_mask_img.to(processor).float()
            #class_label_matrix = groundtruth_label_matrix[:, 1, :, :].to(processor).long()
 
            net_output = tracking_net(input_img)

            obj_loss = objectness_loss(net_output, groundtruth_objectness_mask_img)
            
            #obj_loss = objectness_loss(net_output[:, 0, :, :], objectness_label_matrix)
            #class_loss = classification_loss(net_output[:, 1:, :, :], class_label_matrix)

            #loss_val = OBJECTNESS_LOSS_WEIGHT * obj_loss + CLASSIFICATION_LOSS_WEIGHT * class_loss
            loss_val = OBJECTNESS_LOSS_WEIGHT * obj_loss

            valid_total_loss_list.append(loss_val.item())
            #valid_objectness_loss_list.append(obj_loss.item())
            #valid_CrossEntropy_loss_list.append(class_loss.item())
            
        valid_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(valid_total_loss_list), epoch)
        #valid_writer.add_scalar('Average MSE Loss per Epoch - {}'.format(start_time), np.average(valid_objectness_loss_list), epoch)
        #valid_writer.add_scalar('Average CrossEntropy Loss per Epoch - {}'.format(start_time), np.average(valid_CrossEntropy_loss_list), epoch)

    if np.average(valid_total_loss_list) <= np.average(train_total_loss_list):

        torch.save({'epoech': epoch,
                    'model_state_dict': tracking_net.state_dict(),
                    'training_loss': np.average(train_total_loss_list),
                    'valid_loss': np.average(valid_total_loss_list)},
                    './runs/' + start_time + '/3D_conv_object_tracking_state_dict.pth')
        
        torch.save(tracking_net, './runs/' + start_time + '/3D_conv_object_tracking.pth')

        dummy_input = torch.randn(1, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH).to(processor).float()
        dummy_output = tracking_net(dummy_input)

        torch.onnx.export(
            tracking_net,
            dummy_input,
            './runs/' + start_time + '/3D_conv_object_tracking_state_dict.onnx',
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=13
        )