import torch
import torch.nn as nn
import torch.optim as optim
from torch import device

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ViT_CIFAR_10

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import datetime
import numpy as np

BATCH_SIZE = 64
EPOCH = 100

LEARNING_RATE = 1e-6

THREAD_NUM = 4

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

cifar_10_train_set = torchvision.datasets.CIFAR10(root='./CIFAR_10', train=True, transform=transforms.Compose([transforms.ToTensor()]),download=True)
train_dataloader = DataLoader(dataset=cifar_10_train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

cifar_10_test_set = torchvision.datasets.CIFAR10(root='./CIFAR_10', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
test_dataloader = DataLoader(dataset=cifar_10_test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

image, label = cifar_10_train_set.__getitem__(0)

input_img_channel = image.size(dim=0)
input_img_height = image.size(dim=1)
input_img_width = image.size(dim=2)

print('input_img_channel : {}'.format(input_img_channel))
print('input_img_height : {}'.format(input_img_height))
print('input_img_width : {}'.format(input_img_width))

vit_cifar_10 = ViT_CIFAR_10(input_width=input_img_width, input_height=input_img_height,
                            input_channel=input_img_channel, input_batchsize=BATCH_SIZE,
                            patch_width=4, patch_height=4, output_label_num=10,
                            transformer_embedding_bias=True,
                            transformer_nhead=8,
                            transforemr_internal_feedforward_embedding=128,
                            transformer_dropout=0.1,
                            transformer_activation='gelu',
                            transformer_encoder_layer_num=8,
                            classification_layer_bias=True,
                            device=processor,
                            verbose='low')

vit_cifar_10.to(processor)

optimizer = optim.Adam(vit_cifar_10.parameters(), lr=LEARNING_RATE)

loss_function = nn.CrossEntropyLoss()

### Summary Writer Setup ###
training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/VIT_CIFAR_10_training')
test_writer = SummaryWriter(log_dir='./runs/' + start_time + '/VIT_CIFAR_10_test')

for epoch in range(EPOCH):

    train_loss_list = []
    test_loss_list = []

    print('[EPOCH : {}]'.format(epoch))

    vit_cifar_10.train()

    for batch_idx, (image, label) in enumerate(tqdm(train_dataloader)):

        image = image.to(processor).float()
        label = label.to(processor)

        optimizer.zero_grad()

        output_vector = vit_cifar_10(image)

        loss_val = loss_function(output_vector, label)
        loss_val.backward()

        optimizer.step()

        train_loss_list.append(loss_val.item())

    training_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(train_loss_list), epoch)

    vit_cifar_10.eval()

    with torch.no_grad():
        
        for batch_idx, (image, label) in enumerate(tqdm(test_dataloader)):

            image = image.to(processor).float()
            label = label.to(processor)
            
            output_vector = vit_cifar_10(image)

            loss_val = loss_function(output_vector, label)
            
            test_loss_list.append(loss_val.item())
            
        test_writer.add_scalar('Total Average Loss per Epoch - {}'.format(start_time), np.average(train_loss_list), epoch)





