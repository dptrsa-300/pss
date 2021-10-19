#! /usr/bin/env python
#################################################################################
#     File Name           :     network.py
#     Created By          :     yang
#     Creation Date       :     [2017-01-26 21:54]
#     Last Modified       :     [2017-01-26 22:37]
#     Description         :      
#################################################################################
import lasagne, theano
import theano.tensor as T
from layer import FeatureProjectionLayer, DiagMaskLayer, MeanPooling_1D_Length_Layer, NormalizedLayer
import pickle as cPickle
import numpy as np

import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import Module

class Network(Module):
    def __init__(self):
        self.network = None

    def save_to_file(self, file_name):
        parameter_list = lasagne.layers.get_all_param_values(self.network)
        with open(file_name, "w") as fout:
            cPickle.dump(parameter_list, fout)

    def load_from_file(self, file_name):
        with open(file_name, 'rb') as fin:
            parameter_list = cPickle.load(fin, encoding="bytes")
        lasagne.layers.set_all_param_values(self.network, parameter_list)

class NetworkLen(Network):
    def __init__(self):
        super(NetworkLen, self).__init__()
        self.embedding_func = None

    def build_theano_embedding_function(self, deterministic=None):
        input_tensor = T.tensor3("input_tensor", dtype="float32")
        length_tensor = T.ivector("length_tensor")
        embedding = lasagne.layers.get_output(self.network, {
            self.input_layer: input_tensor,
            self.length_layer: length_tensor
        }, deterministic=deterministic)
        func = theano.function([input_tensor, length_tensor], embedding, updates=None)
        return func

    def get_embedding(self, distance_matrix):
        embedding_size = self.network.output_shape[-1]
        length = distance_matrix.shape[0]
        distance_matrix[range(length), range(length)] = float("inf")
        self.distance_matrix_buf[:, :] = float("inf")
        self.distance_matrix_buf[:length, :length] = distance_matrix[:, :]
        if self.embedding_func is None:
            self.embedding_func = self.build_theano_embedding_function(deterministic=True)
        return self.embedding_func(self.distance_matrix_buf[None, :, :], np.array([length], dtype = 'int32'))[0]

class DeepFold(NetworkLen):
    def __init__(self, max_length=256, projection_level=3):
        self.distance_matrix_buf = np.zeros((max_length, max_length), dtype = 'float32')
        super(DeepFold, self).__init__()
        self.cnn_layers = []

        self.input_layer = lasagne.layers.InputLayer(shape=(None, max_length, max_length))
        self.length_layer = lasagne.layers.InputLayer(shape=(None,))
        feature_layer = FeatureProjectionLayer(self.input_layer, projection_level=projection_level)
        filter_number_list = [128, 256, 512, 512, 512, 398]
        filter_size_list = [12, 4, 4, 4, 4, 4]
        last_output_channels = projection_level
        for filter_number, filter_size in zip(filter_number_list, filter_size_list):
            self.cnn_layers.append(nn.Conv2d(in_channels=last_output_channels,
                                             out_channels=filter_number,
                                             kernel_size=(filter_size, filter_size), # square kernel
                                             padding=int(filter_size / 2 - 1),
                                             stride=(2, 2)))
            last_output_channels = filter_number
            self.cnn_layers.append(lasagne.layers.BatchNorm2d(filter_number))
            self.cnn_layers.append(nn.ReLU(inplace=True))
            self.cnn_layers.append(nn.Dropout(p=0.5))
        
        feature_layer = DiagMaskLayer(feature_layer)
        feature_layer = MeanPooling_1D_Length_Layer(incomings=[feature_layer, self.length_layer],
                                                    factor=2 ** (len(filter_number_list)))
        feature_layer = NormalizedLayer(feature_layer)
        self.network = feature_layer

    def forward(self, x):
        return x

