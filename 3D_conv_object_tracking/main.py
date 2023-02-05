import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader

import h5py
from tqdm import tqdm

from dataset_compiler import dataset_compiler
from dataset import object_tracking_dataset

from model import Object_Tracking_Net

### Hyper Parameter ###
BATCH_SIZE = 32
EPOCH = 100

LEARNING_RATE = 1e-4

### Processor Setup ###
processor = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(processor)
print(processor)

dataset_compiler(dataset_save_path='./dataset.hdf5',
                 image_dataset_path='/home/luwis/ICSL_Project/KITTI_dataset/Object_Video_Tracking/data_tracking_image_2/training/image_02',
                 groundtruth_dataset_path='/home/luwis/ICSL_Project/KITTI_dataset/Object_Video_Tracking/data_tracking_label_2/training/label_02',
                 train_sequence=['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007'],
                 valid_sequence=['0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015'],
                 test_sequence=['0016', '0017', '0018', '0019', '0020'],
                 skip_count=1,
                 verbose='low')

train_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', mode='train', verbose='low')
train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

valid_set = object_tracking_dataset(dataset_save_path='./dataset.hdf5', mode='valid', verbose='low')
valid_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

tracking_net = Object_Tracking_Net(bias=True, verbose='high')
tracking_net.to(processor)

optimizer = optim.Adam(tracking_net.parameters(), lr=LEARNING_RATE)

loss_function = nn.MSELoss()

for epoch in range(EPOCH):

    tracking_net.train()

    for batch_idx, (input_img) in enumerate(tqdm(train_dataloader)):
    # for batch_idx, (input_img, groundtruth_label) in enumerate(tqdm(train_dataloader)):

        input_img = input_img.to(processor).float()
        # groundtruth_label = groundtruth_label.to(processor).float()

        optimizer.zero_grad()

        net_output = tracking_net(input_img)

        # loss_val = loss_function(net_output, groundtruth_label)
        # loss_val.backward()
        
        optimizer.step()

    tracking_net.eval()

    with torch.no_grad():
        
        for batch_idx, (input_img) in enumerate(tqdm(valid_dataloader)):
        # for batch_idx, (input_img, groundtruth_label) in enumerate(tqdm(valid_dataloader)):

            input_img = input_img.to(processor).float()
            # groundtruth_label = groundtruth_label.to(processor).float()
                
            net_output = tracking_net(input_img)

            # loss_val = loss_function(net_output, groundtruth_label)