import torch.nn as nn
from torch.nn.functional import avg_pool2d, dropout, max_pool2d, relu  
import numpy as np
import math
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


#####################################################################################
#######   MNIST models
#######
#####################################################################################

class dnn2(nn.Module):
    def __init__(self, num_classes=10, dataset='mnist') -> None:
        super(dnn2,self).__init__()
        self.input_shape = datasets_in_shape[dataset]
        self.linear1 = nn.Linear(np.prod(self.input_shape),200)
        self.linear2 = nn.Linear(200,num_classes)
        self.flat_op = Flatten() if len(self.input_shape) > 1 else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.flat_op(x)
        out = self.activation((self.linear1(out)))
        out = self.linear2(out)
        return out


class dnn2_tanh(dnn2):
    def __init__(self, num_classes=10, dataset='mnist') -> None:
        super(dnn2_tanh,self).__init__(num_classes=num_classes, dataset=dataset)
        self.activation = nn.Tanh()
        

class dnn4(nn.Module):
    def __init__(self, num_classes=10, dataset='mnist') -> None:
        super(dnn4, self).__init__()
        self.input_shape = datasets_in_shape[dataset]
        self.linear1 = nn.Linear(np.prod(self.input_shape),200)

        self.linear2 = nn.Linear(200,100)
        self.linear3 = nn.Linear(100,100)
        self.linear4 = nn.Linear(100,num_classes) 
        self.flat_op = Flatten() if len(self.input_shape) > 1 else lambda x:x
        
    def forward(self, x):
        out = self.flat_op(x)
        out = nn.ReLU()(self.linear1(out))
        out = nn.ReLU()(self.linear2(out))
        out = nn.ReLU()(self.linear3(out))
        out = self.linear4(out)
        return out

    
class dnn4_tanh(dnn4):
    def __init__(self, num_classes, dataset='mnist') -> None:
        super(dnn4_tanh,self).__init__(num_classes=num_classes,dataset=dataset)
        self.activation = nn.Tanh()
        

class CNN_custom(nn.Module):
    def __init__(self, num_classes=10, dataset='mnist'):
        super(CNN_custom,self).__init__()
        self.input_shape = datasets_in_shape[dataset]
        self.conv1 = nn.Conv2d(self.input_shape[0],32,3,padding=1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(32,32,3, padding=1,stride=2)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,3,padding=1,stride=2)
        flat_shapes = {'mnist':3136,'cifar10':4096}
        self.linear1 = nn.Linear(flat_shapes[dataset],100)
        self.linear2 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = nn.ReLU()(self.conv1(x))
        out = nn.ReLU()(self.conv2(out))
        out = nn.ReLU()(self.conv3(out))
        out = nn.ReLU()(self.conv4(out))
        out = Flatten()(out)
        out = nn.ReLU()(self.linear1(out ))
        out = self.linear2(out)
        return out


#####################################################################################
#######   CIFAR10 models
#######
#####################################################################################


class LeNet(nn.Module):
    def __init__(self, num_classes=10, dataset='cifar10'):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.input_shape = datasets_in_shape[dataset]
        self.conv1 = nn.Conv2d(self.input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        out = nn.ReLU()(self.conv1(x))
        out = max_pool2d(out, 2)
        out = nn.ReLU()(self.conv2(out))
        out = max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = relu(self.fc1(out))
        out = relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ConvNet(nn.Module):
    
    def __init__(self, num_classes=10, dataset='cifar10'):
        super(ConvNet, self).__init__()
        self.input_shape = datasets_in_shape[dataset]
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = relu(self.conv1(x))         # 32*32*48
        x = relu(self.conv2(x))         # 32*32*96
        x = self.pool(x)                # 16*16*96
        x = self.Dropout(x)
        x = relu(self.conv3(x))         # 16*16*192
        x = relu(self.conv4(x))         # 16*16*256
        x = self.pool(x)                # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8 * 8 * 256)     # reshape x
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


''' ResNet18 for CIFAR10 in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlockResNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_planes=2, dataset='cifar10'):
    
    return ResNet(BasicBlockResNet, [2, 2, 2, 2], in_planes=in_planes)


#####################################################################################
#######   CIFAR100 model
#######
#####################################################################################


"""
Code from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
BSD 3-Clause License
Copyright (c) 2017, Andreas Veit
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = dropout(out, p=self.droprate, inplace=False, training=self.training)
        return avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth=40, num_classes=100, growth_rate=40, dataset='cifar100', reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3

        if bottleneck is True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


############ NasNet for CIFAR100 ############

"""nasnet in pytorch

[1] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le

    Learning Transferable Architectures for Scalable Image Recognition
    https://arxiv.org/abs/1707.07012
"""


class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            **kwargs
        )

        self.pointwise = nn.Conv2d(
            input_channels,
            output_channels,
            1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeperableBranch(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(output_channels, output_channels, kernel_size, stride=1, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Fit(nn.Module):
    """Make the cell outputs compatible

    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()

        self.p1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        #make sure there is no information loss
        self.p2 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),   # cropping
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        self.bn = nn.BatchNorm2d(filters)

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(prev_filters, filters, 1),
            nn.BatchNorm2d(filters)
        )

        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x

        #image size does not match
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)

        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)

        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dem_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.block1_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block1_right = nn.Sequential()

        self.block2_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block2_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()

        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.block5_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.block5_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x

        #return transformed x as new x, and original x as prev
        #only prev tensor needs to be modified
        prev = self.fit((x, prev))

        h = self.dem_reduce(x)

        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)

        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x


class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1),
            nn.BatchNorm2d(output_channels)
        )

        #block1
        self.layer1block1_left = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        #block2
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)

        #block3
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        #block5
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels, output_channels, 3, stride=1, padding=1)

        #block4
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))

        h = self.dim_reduce(x)

        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h)
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev)
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev)
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1) + self.layer2block2_right(layer1block2)

        return torch.cat([
            layer1block2,  # https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py line 739
            layer1block3,
            layer2block1,
            layer2block2
        ], 1), x


class NasNetA(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter, class_num=100):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stemfilter, 3, padding=1, bias=False),
            nn.BatchNorm2d(stemfilter)
        )

        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters

        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)

        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 6, class_num)

    def _make_normal(self, block, repeat, output):
        """make normal cell
        Args:
            block: cell type
            repeat: number of repeated normal cell
            output: output filters for each branch in normal cell
        Returns:
            stacked normal cells
        """

        layers = []
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * 6         # concatenate 6 branches

        return layers

    def _make_reduction(self, block, output):
        """make normal cell
        Args:
            block: cell type
            output: output filters for each branch in reduction cell
        Returns:
            reduction cell
        """

        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4     # stack for 4 branches

        return reduction

    def _make_layers(self, repeat_cell_num, reduction_num):

        layers = []
        for i in range(reduction_num):

            layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))

        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


supported_datasets = ['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'imagenet']
datasets_in_shape = { 'mnist': (1,28,28), 'cifar10': (3,32,32), 'cifar100': (3,32,32), 'imagenet': (3,224,224) }
datasets_means = {'mnist': 0,
                  'cifar10': (0.4914, 0.4822, 0.4465),
                  'cifar100': [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0], 
                  'imagenet': (0.485, 0.456, 0.406)}
datasets_stds = {'mnist': 1,
                 'cifar10': (0.2023, 0.1994, 0.2010),
                 'cifar100': [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0],
                 'imagenet': (0.229, 0.224, 0.225)}
datasets_supp_archs = {'mnist': {'dnn2':dnn2,'dnn_2':dnn2,'dnn_4':dnn4,'dnn4':dnn4,'cnn_custom':CNN_custom},
                       'cifar10': {'lenet':LeNet,'convnet':ConvNet,'dnn2':dnn2,'resnet18':ResNet18},
                       'cifar100':{'densenet':DenseNet3}}
datasets_default_arch = {'mnist': 'dnn2', 'cifar10': 'convnet', 'cifar100': 'densenet', 'imagenet': 'resnet18'}
defaults_datasets = ['mnist','cifar10','cifar100','imagenet']
cifar_datasets = ['cifar10','cifar100']
