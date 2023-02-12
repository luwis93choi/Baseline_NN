import torch

import torch.nn as nn

import numpy as np

class ViT_CIFAR_10(nn.Module):
    
    def __init__(self, input_width=32,
                       input_height=32,
                       input_channel=3,
                       input_batchsize=1,
                       patch_width=4,
                       patch_height=4,
                       output_label_num=10,
                       transformer_embedding_bias=True,
                       transformer_nhead=4,
                       transforemr_internal_feedforward_embedding=2048,
                       transformer_dropout=0.1,
                       transformer_activation='gelu',
                       transformer_encoder_layer_num=8,
                       classification_layer_bias=True,
                       device='',
                       verbose='low'):

        super(ViT_CIFAR_10, self).__init__()

        self.verbose = verbose

        if (input_width % patch_width) != 0:
            raise Exception('Input width is not divisible by patch width')

        if (input_height % patch_height) != 0:
            raise Exception('Input height is not divisible by patch height')

        self.total_patch_num = (input_width//patch_width) * (input_height//patch_height)

        if (self.total_patch_num % transformer_nhead) != 0:
            raise Exception('Total patch number is not divisible by number of transformer heads / Patches have to be equally distributed among transformer heads')

        self.embedding_size = patch_width * patch_height * input_channel

        self.cls_embedding = nn.Parameter(torch.zeros(input_batchsize, 1, self.embedding_size))

        # Positional Embedding Generation (Reference : https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
        self.single_positional_embedding = torch.ones(1, self.total_patch_num + 1, self.embedding_size)

        for i in range(self.total_patch_num + 1):
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

        self.classification_layer_1 = nn.Linear(in_features=self.embedding_size, out_features=output_label_num, bias=classification_layer_bias)
        
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, input_img):

        self.local_print('input_img : {}'.format(input_img.size()), level='high')

        patch_embeddings = self.patch_embedder_layer(input_img)
        self.local_print('patch_embeddings : {}'.format(patch_embeddings.size()), level='high')

        flatten_patch_embeddings = self.flatten_layer(patch_embeddings)
        self.local_print('flatten_patch_embeddings : {}'.format(flatten_patch_embeddings.size()), level='high')

        flatten_patch_embeddings = torch.permute(flatten_patch_embeddings, (0, 2, 1))
        self.local_print('flatten_patch_embeddings after dim swap : {}'.format(flatten_patch_embeddings.size()), level='high')

        flatten_patch_embeddings_with_CLS = torch.cat((self.cls_embedding, flatten_patch_embeddings), dim=1)
        self.local_print('flatten_patch_embeddings_with_CLS : {}'.format(flatten_patch_embeddings_with_CLS.size()), level='high')

        self.local_print('self.positional_embedding : {}'.format(self.positional_embedding.size()), level='high')

        patch_embeddings_added_with_positional_embeddings = flatten_patch_embeddings_with_CLS + self.positional_embedding
        self.local_print('patch_embeddings_added_with_positional_embeddings : {}'.format(patch_embeddings_added_with_positional_embeddings.size()), level='high')

        transformer_encoder_output = self.transformer_encoder(patch_embeddings_added_with_positional_embeddings)
        self.local_print('transformer_encoder_output : {}'.format(transformer_encoder_output.size()), level='high')

        cls_patch_output = transformer_encoder_output[:, 0, :]
        self.local_print('cls_patch_output : {}'.format(cls_patch_output.size()), level='high')

        classification_output_1 = self.classification_layer_1(cls_patch_output)
        self.local_print('classification_output_1 : {}'.format(classification_output_1.size()), level='high')

        self.local_print('--------------------------', level='high')

        return classification_output_1

    def local_print(self, sen, level='low'):

        if self.verbose == 'high': print(sen)
        elif self.verbose == 'low':
            if level == 'low': print(sen)
