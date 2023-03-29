import functools

import torch.nn as nn
from torchvision import models

CONV3_3_IN_VGG_19 = models.vgg19(pretrained=True).features[:15]

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        # we should never set track_running_stats to True in InstanceNorm
        # because it behaves differently in training and testing mode
        norm_layer = functools.partial(nn.InstanceNorm2d, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class ResNetBlock(nn.Module):
    """ResNet block"""

    def __init__(self, dim, norm_layer, padding_type, use_dropout, use_bias):
        super(ResNetBlock, self).__init__()

        sequence = list()
        padding = self._chose_padding_type(padding_type, sequence)

        sequence += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=padding, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def _chose_padding_type(self, padding_type, sequence):
        padding = 0
        if padding_type == 'reflect':
            sequence += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            sequence += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return padding

    def forward(self, x):
        out = x + self.model(x)
        return out
