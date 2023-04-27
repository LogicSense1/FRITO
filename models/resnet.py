import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F


from ba3l.ingredients.ingredient import Ingredient

model_ing = Ingredient("resnet")

model_ing.add_config(instance_cmd="get_model")


class ResnetLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        strides=1,
        learn_bn: bool = True,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        self.norm_layer = nn.BatchNorm2d(num_features=in_channels, affine=learn_bn)
        if use_relu:
            self.relu = nn.ReLU()
        # missing: kernel_initializer='he_normal', kernel_regularizer=l2(wd)
        # not using: padding='same'
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')  # fix using kaiming normal instead of kaiming uniform

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_layer(x)
        if hasattr(self, 'relu'):
            x = self.relu(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        down_sample: bool = False,
    ) -> None:
        super().__init__()
        strides = 1 if not down_sample else [1,2]
        self.down_sample = down_sample
        if down_sample:
            self.avg_pooling = nn.AvgPool2d(kernel_size=(3,3), stride=[1,2])    # missing: padding='same'
        self.res1 = ResnetLayer(num_filters if not down_sample else num_filters//2, num_filters, strides=strides, learn_bn=False, use_relu=True)
        self.res2 = ResnetLayer(num_filters, num_filters, strides=1, learn_bn=False, use_relu=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if not self.down_sample:
            x = F.pad(x, (2,2,2,2))
        else:
            x = F.pad(x, (2,2,1,1))
        x = self.res1(x)
        x = self.res2(x)    # 到这里channel数目是与keras一致的，但不知道序列长度是否一致？
        if self.down_sample:
            residual = self.avg_pooling(residual)
            residual = torch.cat([residual, torch.zeros_like(residual)], dim=1)
        x = x + residual
        return x


class MyResnetClassifier(nn.Module):
    def __init__(self, input_shape,
            input_channels,
            num_filters,
            num_res_blocks,
            num_labels
        ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        res_layers1 = []
        res_layers2 = []
        res_layers1.append(ResnetLayer(input_channels, num_filters, strides=[1,2], learn_bn=True, use_relu=False))
        res_layers2.append(ResnetLayer(input_channels, num_filters, strides=[1,2], learn_bn=True, use_relu=False))
        self.num_labels = num_labels

        for stack in range(4):
            for res_block in range(self.num_res_blocks):
                down_sample = stack > 0 and res_block == 0
                res_layers1.append(ResnetBlock(num_filters, down_sample=down_sample))
                res_layers2.append(ResnetBlock(num_filters, down_sample=down_sample))
            num_filters *= 2

        self.res_layers1 = nn.Sequential(*res_layers1)
        self.res_layers2 = nn.Sequential(*res_layers2)

        self.final_res_layer = nn.Sequential(
            ResnetLayer(num_filters // 2, 2 * num_filters, kernel_size=1, strides=1, learn_bn=False, use_relu=True),
            ResnetLayer(2 * num_filters, num_labels, kernel_size=1, strides=1, learn_bn=False, use_relu=False),
        )
        self.norm_layer = nn.BatchNorm2d(num_features=num_labels, affine=False)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1)) # fix max pooling -> avg pooling
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, log_mel_spec):
        # (batch, num_channel, num_freq_bin, num_time_bin)
        x = log_mel_spec
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError('Input tensor must have 3 or 4 dimensions.')
        assert x.shape[2] == 128

        # if self.training and self.mixup:
        #     x, labels = self._mixup(x, labels)

        split1 = x[:, :, 0:64, :]
        split2 = x[:, :, 64:128, :]
        residual_path1 = self.res_layers1(split1)
        residual_path2 = self.res_layers2(split2)
        # (b, c, h, w)
        residual_path = torch.cat([residual_path1, residual_path2], dim=2)  # fix: dim=1 -> dim=2
        output_path = self.final_res_layer(residual_path)
        output_path = self.norm_layer(output_path)
        logits = self.global_avg_pooling(output_path).flatten(start_dim=1)
        return logits


@model_ing.command
def get_model(
            input_shape=128,
            input_channels=1,
            num_filters=24,
            num_res_blocks=2,
            num_labels=10
        ):
    model = MyResnetClassifier(input_shape,
        input_channels,
        num_filters,
        num_res_blocks,
        num_labels)
    return model