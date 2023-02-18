import torch

import torch.nn as nn

import numpy as np

class residual_block(nn.Module):
        
    def __init__(self, block_type='1',
                       in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0,
                       bias=True):

        super(residual_block, self).__init__()

        self.block_type = block_type

        # Type 1 Residual : No Channel Reduction
        if self.block_type == '1':

            self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.batchNorm_1 = nn.BatchNorm2d(num_features=out_channels)
            self.activation_1 = nn.ReLU()

            self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.batchNorm_2 = nn.BatchNorm2d(num_features=out_channels)
            self.activation_2 = nn.ReLU()

        # Type 2 Residual : Channel Expansion at Residual Path
        elif self.block_type == '2':
            
            self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.batchNorm_1 = nn.BatchNorm2d(num_features=in_channels)
            self.activation_1 = nn.ReLU()

            self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.batchNorm_2 = nn.BatchNorm2d(num_features=out_channels)
            self.activation_2 = nn.ReLU()

            # Channel Reduction with 1x1 Conv
            self.residual_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              bias=bias)

    def forward(self, input_data):

        if self.block_type == '1':

            conv_1_out = self.activation_1(self.batchNorm_1(self.conv_1(input_data)))
            conv_2_out = self.activation_2(self.batchNorm_2(self.conv_2(conv_1_out)))

            residual_out = input_data

            final_out = conv_2_out + residual_out

            return final_out

        elif self.block_type == '2':

            conv_1_out = self.activation_1(self.batchNorm_1(self.conv_1(input_data)))
            conv_2_out = self.activation_2(self.batchNorm_2(self.conv_2(conv_1_out)))

            residual_out = self.residual_channel(input_data)

            final_out = conv_2_out + residual_out

            return final_out

class Object_Tracking_VIT(nn.Module):
    
    def __init__(self, input_width=32,
                       input_height=32,
                       input_channel=3,
                       reduce_channel=False,
                       input_batchsize=1,
                       patch_width=4,
                       patch_height=4,
                       transformer_embedding_bias=True,
                       transformer_nhead=4,
                       transforemr_internal_feedforward_embedding=2048,
                       transformer_dropout=0.1,
                       transformer_activation='gelu',
                       transformer_encoder_layer_num=8,
                       classification_layer_bias=True,
                       device='',
                       verbose='low'):

        super(Object_Tracking_VIT, self).__init__()

        self.verbose = verbose

        if (input_width % patch_width) != 0:
            raise Exception('Input width is not divisible by patch width')

        if (input_height % patch_height) != 0:
            raise Exception('Input height is not divisible by patch height')

        self.total_patch_num = (input_width//patch_width) * (input_height//patch_height)

        if (self.total_patch_num % transformer_nhead) != 0:
            raise Exception('Total patch number is not divisible by number of transformer heads / Patches have to be equally distributed among transformer heads')

        if reduce_channel == False:
            self.embedding_size = patch_width * patch_height * input_channel
        elif reduce_channel == True:
            self.embedding_size = patch_width * patch_height

        # Positional Embedding Generation (Reference : https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
        self.single_positional_embedding = torch.ones(1, self.total_patch_num, self.embedding_size)

        for i in range(self.total_patch_num):
            for j in range(self.embedding_size):
                self.single_positional_embedding[0][i][j] = np.sin(i / (10000 ** (j/self.embedding_size))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - i) / self.embedding_size)))

        self.positional_embedding = nn.Parameter(self.single_positional_embedding.repeat(input_batchsize, 1, 1))

        #self.positional_embedding = nn.Parameter(torch.zeros(input_batchsize, self.total_patch_num + 1, self.embedding_size))

        self.patch_embedder_layer = nn.Conv2d(in_channels=input_channel, out_channels=self.embedding_size, 
                                              kernel_size=(patch_height, patch_width), 
                                              stride=(patch_height, patch_width), padding=0, bias=transformer_embedding_bias)

        self.flatten_layer = nn.Flatten(start_dim=2)

        unit_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size,
                                                                    nhead=transformer_nhead,
                                                                    dim_feedforward=transforemr_internal_feedforward_embedding,
                                                                    dropout=transformer_dropout,
                                                                    activation=transformer_activation,
                                                                    batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=unit_transformer_encoder_layer,
                                                         num_layers=transformer_encoder_layer_num)

    def forward(self, input_img):

        self.local_print('input_img : {}'.format(input_img.size()), level='high')

        patch_embeddings = self.patch_embedder_layer(input_img)
        self.local_print('patch_embeddings : {}'.format(patch_embeddings.size()), level='high')

        flatten_patch_embeddings = self.flatten_layer(patch_embeddings)
        self.local_print('flatten_patch_embeddings : {}'.format(flatten_patch_embeddings.size()), level='high')

        flatten_patch_embeddings = torch.permute(flatten_patch_embeddings, (0, 2, 1))
        self.local_print('flatten_patch_embeddings after dim swap : {}'.format(flatten_patch_embeddings.size()), level='high')

        self.local_print('self.positional_embedding : {}'.format(self.positional_embedding.size()), level='high')

        patch_embeddings_added_with_positional_embeddings = flatten_patch_embeddings + self.positional_embedding
        self.local_print('patch_embeddings_added_with_positional_embeddings : {}'.format(patch_embeddings_added_with_positional_embeddings.size()), level='high')

        transformer_encoder_output = self.transformer_encoder(patch_embeddings_added_with_positional_embeddings)
        self.local_print('transformer_encoder_output : {}'.format(transformer_encoder_output.size()), level='high')

        self.local_print('--------------------------', level='high')

        return transformer_encoder_output

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)

            
class Object_Tracking_UNet(nn.Module):

    def __init__(self, bias=True, dropout_prob=0.1, verbose='low'):

        super(Object_Tracking_UNet, self).__init__()

        self.verbose = verbose

        def Conv_2D_Block(in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=True, 
                          pooling_type='max', pooling_kernel_size=2):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.LeakyReLU()]

            if pooling_type == 'max':
                layers += [nn.MaxPool2d(kernel_size=pooling_kernel_size)]
            elif pooling_type == 'avg':
                layers += [nn.AvgPool2d(kernel_size=pooling_kernel_size)]

            layer_module = nn.Sequential(*layers)

            return layer_module
        
        # Encoder Layer 1 : 2D Convolution + Max Pooling
        self.encoder1_1 = Conv_2D_Block(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.encoder1_2 = Conv_2D_Block(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_enc_1 = nn.Dropout2d(p=dropout_prob)

        self.encoder1_pool = nn.MaxPool2d(kernel_size=2)
        
        # Encoder Layer 2 : 2D Convolution + Max Pooling
        self.encoder2_1 = Conv_2D_Block(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.encoder2_2 = Conv_2D_Block(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_enc_2 = nn.Dropout2d(p=dropout_prob)

        self.encoder2_pool = nn.MaxPool2d(kernel_size=2)

        # Encoder Layer 3 : 2D Convolution + Max Pooling
        self.encoder3_1 = Conv_2D_Block(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.encoder3_2 = Conv_2D_Block(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_enc_3 = nn.Dropout2d(p=dropout_prob)

        self.encoder3_pool = nn.MaxPool2d(kernel_size=2)
        
        # Encoder Layer 4 : 2D Convolution + Max Pooling
        self.encoder4_1 = Conv_2D_Block(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.encoder4_2 = Conv_2D_Block(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_enc_4 = nn.Dropout2d(p=dropout_prob)

        self.encoder4_pool = nn.MaxPool2d(kernel_size=2)
        
        # Encoder Layer 5 : 2D Convolution
        self.encoder5_1 = Conv_2D_Block(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)

        # ---------------------------------------------------------------------------------------------------------------

        self.dropout_enc_5 = nn.Dropout2d(p=dropout_prob)
        
        # ---------------------------------------------------------------------------------------------------------------

        # Decoder Layer 5 : 2D Convolution
        self.decoder5_1 = Conv_2D_Block(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)

        # Decoder Layer 4 : 2D Transposed Convolution + Convolution
        self.unpool_4 = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        self.decoder4_1 = Conv_2D_Block(in_channels=(48 + 48), out_channels=48, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.decoder4_2 = Conv_2D_Block(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_dec_4 = nn.Dropout2d(p=dropout_prob)
        
        # Decoder Layer 3 : 2D Transposed Convolution + Convolution
        self.unpool_3 = nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        self.decoder3_1 = Conv_2D_Block(in_channels=(24 + 24), out_channels=24, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.decoder3_2 = Conv_2D_Block(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_dec_3 = nn.Dropout2d(p=dropout_prob)
        
        # Decoder Layer 2 : 2D Transposed Convolution + Convolution
        self.unpool_2 = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        self.decoder2_1 = Conv_2D_Block(in_channels=(12 + 12), out_channels=12, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.decoder2_2 = Conv_2D_Block(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_dec_2 = nn.Dropout2d(p=dropout_prob)
        
        # Decoder Layer 1 : 2D Transposed Convolution + Convolution
        self.unpool_1 = nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        self.decoder1_1 = Conv_2D_Block(in_channels=(6 + 6), out_channels=6, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        self.decoder1_2 = Conv_2D_Block(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True,
                                        pooling_type='none', pooling_kernel_size=2)
        
        self.dropout_dec_1 = nn.Dropout2d(p=dropout_prob)
        
        # 1x1 Conv-based 1 Channel Mask Image Reconstruction
        self.conv_fc = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_img):

        self.local_print('input_img : {}'.format(input_img.size()), level='high')

        # ----------------------------- Encoding Path -------------------------------------------
        # Encoder Layer 1 : 2D Convolution  + Max Pooling
        out_encoder1_1 = self.encoder1_1(input_img)
        self.local_print('out_encoder1_1 : {}'.format(out_encoder1_1.size()), level='high')

        out_encoder1_2 = self.encoder1_2(out_encoder1_1)
        self.local_print('out_encoder1_2 : {}'.format(out_encoder1_2.size()), level='high')

        out_encoder1_2 = self.dropout_enc_1(out_encoder1_2)
        self.local_print('out_encoder1_2 : {}'.format(out_encoder1_2.size()), level='high')

        out_pool_1 = self.encoder1_pool(out_encoder1_2)
        self.local_print('out_pool_1 : {}'.format(out_pool_1.size()), level='high')

        # Encoder Layer 2 : 2D Convolution + Max Pooling 
        out_encoder2_1 = self.encoder2_1(out_pool_1)
        self.local_print('out_encoder2_1 : {}'.format(out_encoder2_1.size()), level='high')

        out_encoder2_2 = self.encoder2_2(out_encoder2_1)
        self.local_print('out_encoder2_2 : {}'.format(out_encoder2_2.size()), level='high')

        out_encoder2_2 = self.dropout_enc_2(out_encoder2_2)
        self.local_print('out_encoder2_2 : {}'.format(out_encoder2_2.size()), level='high')

        out_pool_2 = self.encoder2_pool(out_encoder2_2)
        self.local_print('out_pool_2 : {}'.format(out_pool_2.size()), level='high')

        # Encoder Layer 3 : 2D Convolution + Max Pooling 
        out_encoder3_1 = self.encoder3_1(out_pool_2)
        self.local_print('out_encoder3_1 : {}'.format(out_encoder3_1.size()), level='high')

        out_encoder3_2 = self.encoder3_2(out_encoder3_1)
        self.local_print('out_encoder3_2 : {}'.format(out_encoder3_2.size()), level='high')

        out_encoder3_2 = self.dropout_enc_3(out_encoder3_2)
        self.local_print('out_encoder3_2 : {}'.format(out_encoder3_2.size()), level='high')

        out_pool_3 = self.encoder3_pool(out_encoder3_2)
        self.local_print('out_pool_3 : {}'.format(out_pool_3.size()), level='high')

        # Encoder Layer 4 : 2D Convolution + Max Pooling 
        out_encoder4_1 = self.encoder4_1(out_pool_3)
        self.local_print('out_encoder4_1 : {}'.format(out_encoder4_1.size()), level='high')

        out_encoder4_2 = self.encoder4_2(out_encoder4_1)
        self.local_print('out_encoder4_2 : {}'.format(out_encoder4_2.size()), level='high')

        out_encoder4_2 = self.dropout_enc_4(out_encoder4_2)
        self.local_print('out_encoder4_2 : {}'.format(out_encoder4_2.size()), level='high')

        out_pool_4 = self.encoder4_pool(out_encoder4_2)
        self.local_print('out_pool_4 : {}'.format(out_pool_4.size()), level='high')

        # Encoder Layer 5 : 2D Convolution 
        out_encoder5_1 = self.encoder5_1(out_pool_4)
        self.local_print('out_encoder5_1 : {}'.format(out_encoder5_1.size()), level='high')

        # ---------------------------------------------------------------------------------------

        out_encoder5_1 = self.dropout_enc_5(out_encoder5_1)
        self.local_print('out_encoder5_1 : {}'.format(out_encoder5_1.size()), level='high')

        # ----------------------------- Decoding Path -------------------------------------------
        # Decoder Layer 5 : 2D Convolution 
        out_decoder5_1 = self.decoder5_1(out_encoder5_1)
        self.local_print('out_decoder5_1 : {}'.format(out_decoder5_1.size()), level='high')

        # Decoder Layer 4 : 2D Transposed Convolution + Convolution
        out_unpool_4 = self.unpool_4(out_decoder5_1)
        self.local_print('out_unpool_4 : {}'.format(out_unpool_4.size()), level='high')

        out_decoder4_1 = self.decoder4_1(torch.cat((out_unpool_4, out_encoder4_2), dim=1))
        self.local_print('out_decoder4_1 : {}'.format(out_decoder4_1.size()), level='high')

        out_decoder4_2 = self.decoder4_2(out_decoder4_1)
        self.local_print('out_decoder4_2 : {}'.format(out_decoder4_2.size()), level='high')

        out_decoder4_2 = self.dropout_dec_4(out_decoder4_2)
        self.local_print('out_decoder4_2 : {}'.format(out_decoder4_2.size()), level='high')

        # Decoder Layer 3 : 2D Transposed Convolution + Convolution
        out_unpool_3 = self.unpool_3(out_decoder4_2)
        self.local_print('out_unpool_3 : {}'.format(out_unpool_3.size()), level='high')

        out_decoder3_1 = self.decoder3_1(torch.cat((out_unpool_3, out_encoder3_2), dim=1))
        self.local_print('out_decoder3_1 : {}'.format(out_decoder3_1.size()), level='high')

        out_decoder3_2 = self.decoder3_2(out_decoder3_1)
        self.local_print('out_decoder3_2 : {}'.format(out_decoder3_2.size()), level='high')

        out_decoder3_2 = self.dropout_dec_3(out_decoder3_2)
        self.local_print('out_decoder3_2 : {}'.format(out_decoder3_2.size()), level='high')

        # Decoder Layer 2 : 2D Transposed Convolution + Convolution
        out_unpool_2 = self.unpool_2(out_decoder3_2)
        self.local_print('out_unpool_2 : {}'.format(out_unpool_2.size()), level='high')

        out_decoder2_1 = self.decoder2_1(torch.cat((out_unpool_2, out_encoder2_2), dim=1))
        self.local_print('out_decoder2_1 : {}'.format(out_decoder2_1.size()), level='high')

        out_decoder2_2 = self.decoder2_2(out_decoder2_1)
        self.local_print('out_decoder2_2 : {}'.format(out_decoder2_2.size()), level='high')

        out_decoder2_2 = self.dropout_dec_2(out_decoder2_2)
        self.local_print('out_decoder2_2 : {}'.format(out_decoder2_2.size()), level='high')

        # Decoder Layer 1 : 2D Transposed Convolution + Convolution
        out_unpool_1 = self.unpool_1(out_decoder2_2)
        self.local_print('out_unpool_1 : {}'.format(out_unpool_1.size()), level='high')

        out_decoder1_1 = self.decoder1_1(torch.cat((out_unpool_1, out_encoder1_2), dim=1))
        self.local_print('out_decoder1_1 : {}'.format(out_decoder1_1.size()), level='high')

        out_decoder1_2 = self.decoder1_2(out_decoder1_1)
        self.local_print('out_decoder1_2 : {}'.format(out_decoder1_2.size()), level='high')

        out_decoder1_2 = self.dropout_dec_1(out_decoder1_2)
        self.local_print('out_decoder1_2 : {}'.format(out_decoder1_2.size()), level='high')

        # 1x1 Conv-based 1 Channel Mask Image Reconstruction
        out_conv_fc = self.conv_fc(out_decoder1_2)
        self.local_print('out_conv_fc : {}'.format(out_conv_fc.size()), level='high')

        out_conv_fc = torch.squeeze(out_conv_fc, dim=1)
        self.local_print('out_conv_fc : {}'.format(out_conv_fc.size()), level='high')

        return out_conv_fc

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)