import copy

import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.load_parameter import *
import cv2

class limitRangeLayer(nn.Module):
    def __init__(self, target_min=0, target_max=255):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        x02 = torch.sigmoid(x / 100)
        scale = self.target_max - self.target_min
        return x02 * scale + self.target_min


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_of_filters = 12
        filename = '2d_x2.mat'
        (weights, biases, prelu) = load_parameter_FSRCNN(filename)

        first2dConv = nn.Conv2d(1, 32, 5, padding=2)

        with torch.no_grad():
            ini = weights[0].reshape((32, 1, 5, 5))
            first2dConv.weight.data = torch.FloatTensor(ini)
            first2dConv.bias.data = torch.FloatTensor(biases[0])
        print("finish setting weight")

        twodConvBlock1BluePrint = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, num_of_filters, 1),
            nn.ReLU(),
        )
        twodConvBlock2BluePrint = nn.Sequential(
            nn.Conv2d(num_of_filters, num_of_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_of_filters, num_of_filters, 3, padding=1)
        )
        twodConvBlock3BluePrint = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_of_filters, 32, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.ReLU()
        )
        self.first2dConv1 = copy.deepcopy(first2dConv)
        self.first2dConv2 = copy.deepcopy(first2dConv)
        self.first2dConv3 = copy.deepcopy(first2dConv)
        self.first2dConv4 = copy.deepcopy(first2dConv)
        self.first2dConv5 = copy.deepcopy(first2dConv)
        self.twodConvBlock11 = copy.deepcopy(twodConvBlock1BluePrint)
        self.twodConvBlock12 = copy.deepcopy(twodConvBlock2BluePrint)
        self.twodConvBlock13 = copy.deepcopy(twodConvBlock3BluePrint)
        self.twodConvBlock21 = copy.deepcopy(twodConvBlock1BluePrint)
        self.twodConvBlock22 = copy.deepcopy(twodConvBlock2BluePrint)
        self.twodConvBlock23 = copy.deepcopy(twodConvBlock3BluePrint)
        self.twodConvBlock31 = copy.deepcopy(twodConvBlock1BluePrint)
        self.twodConvBlock32 = copy.deepcopy(twodConvBlock2BluePrint)
        self.twodConvBlock33 = copy.deepcopy(twodConvBlock3BluePrint)
        self.twodConvBlock41 = copy.deepcopy(twodConvBlock1BluePrint)
        self.twodConvBlock42 = copy.deepcopy(twodConvBlock2BluePrint)
        self.twodConvBlock43 = copy.deepcopy(twodConvBlock3BluePrint)
        self.twodConvBlock51 = copy.deepcopy(twodConvBlock1BluePrint)
        self.twodConvBlock52 = copy.deepcopy(twodConvBlock2BluePrint)
        self.twodConvBlock53 = copy.deepcopy(twodConvBlock3BluePrint)

        self.ResReLUList = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]

        self.storeTwodConvBlock = [self.twodConvBlock11, self.twodConvBlock12, self.twodConvBlock13,
                                   self.twodConvBlock21, self.twodConvBlock22, self.twodConvBlock23,
                                   self.twodConvBlock31, self.twodConvBlock32, self.twodConvBlock33,
                                   self.twodConvBlock41, self.twodConvBlock42, self.twodConvBlock43,
                                   self.twodConvBlock51, self.twodConvBlock52, self.twodConvBlock53]
        self.storeFirst2dConv = [self.first2dConv1, self.first2dConv2, self.first2dConv3, self.first2dConv4,
                                 self.first2dConv5]
        self.threedConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 5, 3), padding=(2, 2, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 8, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.ReLU()
        )

        self.threedConvBlock2 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.threedConvBlock3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(8, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.ReLU(),
            # reducing size of image
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3)),
            nn.ReLU()
        )
        self.output2dlayer = nn.Conv2d(32, 1, kernel_size=(5, 5), padding=2)
        self.output2dlayer2 = nn.Conv2d(1, 1, kernel_size=(5, 5), padding=2)
        self.limitRangeLayer = limitRangeLayer()

    def forward(self, input):
        batch_size, channel, depth, height, width = input.shape
        save_interim_output = []
        number_of_recursion_2d = 5
        number_of_recursion_3d = 5
        input = input.float()
        for i in range(5):
            x = input[:, :, i, :, :]

            # res across net
            # if i == 2:
            #     res_across_net_copy = copy.deepcopy(x)
            #     res_across_net_copy = np.array(res_across_net_copy)
            #     res_across_net = []
            #     for j in range(batch_size):
            #         print(height, width)
            #         print(res_across_net_copy[j,0,:,:].shape)
            #         tmp = res_across_net_copy[j,0,:,:]
            #         reshape_tmp = cv2.resize(tmp, (height * 2, width * 2))
            #         print("shape of reshape_tmp", reshape_tmp.shape)
            #         reshape_tmp = np.expand_dims(reshape_tmp, axis=0)
            #         reshape_tmp = np.expand_dims(reshape_tmp, axis=0)
            #         reshape_tmp = reshape_tmp[:, :, 2:2 * height - 2, 2:2 * width - 2]
            #         print("shape of reshape_tmp", reshape_tmp.shape)
            #         res_across_net.append(reshape_tmp)
            #         print("resr across net", np.array(res_across_net).shape)
            #     res_across_net = torch.tensor(res_across_net)
            #     print(res_across_net.shape)
            #     # to be added to the end of the network

            x = self.storeFirst2dConv[i](x)
            x = self.storeTwodConvBlock[3 * i](x)
            # recursive res block, with res not from recursion
            if i > 1 and i < 4:
                for k in range(number_of_recursion_2d-1):
                    x = self.storeTwodConvBlock[3 * i + 1](x) + x
                    x = self.ResReLUList[i](x)
            # second_red2d = x
            # if i == 2:
            #     for p in range(number_of_recursion_2d):
            #         x = self.storeTwodConvBlock[3 * i + 1](x) + x
            # x = x + second_red2d
            x = self.storeTwodConvBlock[3 * i + 2](x)
            # adding one dimension on the input for cat to
            x = x[:, :, None, :, :]
            save_interim_output.append(x)

        interim_output = torch.cat(save_interim_output, dim=2)
        x = self.threedConvBlock1(interim_output)
        # adding recursion feature to supplement
        for k in range(number_of_recursion_3d):
            res3d = x
            x = self.threedConvBlock2(x) + res3d
        x = self.threedConvBlock3(x)
        output = self.output2dlayer(torch.reshape(x, (batch_size, 32, height * 2 - 2 * 2, width * 2 - 2 * 2)))
        # print("Shape of res_across net",res_across_net.shape)
        # output = output + res_across_net[:, :, 0, :, :]
        #output = self.output2dlayer2(output)
        output = self.limitRangeLayer(output)
        return output
