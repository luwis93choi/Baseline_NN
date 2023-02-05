import torch

import torch.nn as nn

class Object_Tracking_Net(nn.Module):

    def __init__(self, bias=True, verbose='low'):

        super(Object_Tracking_Net, self).__init__()

        self.verbose = verbose

        # 3D Convolution Layer 1
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=12, kernel_size=(2, 7, 7), stride=2, padding=0, bias=bias)
        self.batchnorm3d_1 = nn.BatchNorm3d(num_features=12)
        self.activation_1 = nn.ReLU()
        
        # 3D Convolution Layer 2
        self.conv3d_2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 5, 5), stride=2, padding=0, bias=bias)
        self.batchnorm3d_2 = nn.BatchNorm3d(num_features=24)
        self.activation_2 = nn.ReLU()

        # 3D Convolution Layer 3
        self.conv3d_3 = nn.Conv3d(in_channels=24, out_channels=36, kernel_size=(1, 3, 3), stride=2, padding=0, bias=bias)
        self.batchnorm3d_3 = nn.BatchNorm3d(num_features=36)
        self.activation_3 = nn.ReLU()

    def forward(self, input_img):

        self.local_print('input_img : {}'.format(input_img.size()), level='high')

        # 3D Convolution Layer 1
        out_conv3d_1 = self.conv3d_1(input_img)
        self.local_print('out_conv3d_1 : {}'.format(out_conv3d_1.size()), level='high')
        
        out_batchnorm3d_1 = self.batchnorm3d_1(out_conv3d_1)
        self.local_print('out_batchnorm3d_1 : {}'.format(out_batchnorm3d_1.size()), level='high')
        
        out_activation_1 = self.activation_1(out_batchnorm3d_1)
        self.local_print('out_activation_1 : {}'.format(out_activation_1.size()), level='high')

        # 3D Convolution Layer 2
        out_conv3d_2 = self.conv3d_2(out_activation_1)
        self.local_print('out_conv3d_2 : {}'.format(out_conv3d_2.size()), level='high')
        
        out_batchnorm3d_2 = self.batchnorm3d_2(out_conv3d_2)
        self.local_print('out_batchnorm3d_2 : {}'.format(out_batchnorm3d_2.size()), level='high')
        
        out_activation_2 = self.activation_2(out_batchnorm3d_2)
        self.local_print('out_activation_2 : {}'.format(out_activation_2.size()), level='high')

        # 3D Convolution Layer 3
        out_conv3d_3 = self.conv3d_3(out_activation_2)
        self.local_print('out_conv3d_3 : {}'.format(out_conv3d_3.size()), level='high')
        
        out_batchnorm3d_3 = self.batchnorm3d_3(out_conv3d_3)
        self.local_print('out_batchnorm3d_3 : {}'.format(out_batchnorm3d_3.size()), level='high')
        
        out_activation_3 = self.activation_3(out_batchnorm3d_3)
        self.local_print('out_activation_3 : {}'.format(out_activation_3.size()), level='high')

        # return output

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)