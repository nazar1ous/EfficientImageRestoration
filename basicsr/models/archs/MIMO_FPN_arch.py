import time

import numpy as np
import torch
import torch.nn as nn
import math

import torch.nn as nn
from basicsr.models.archs.MIMO_Unet_arch import *

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# def is_frozen_model(model):
#     print("is frozen check")
#     l = []
#     for layer in model.children():
#         for parameter in layer.parameters():
#             l.append(parameter.requires_grad)
#     print(l)
#     return not all(l)

def is_unfrozen_model(model):
    # print("is unfrozen check")
    l = []
    for layer in model.children():
        for parameter in layer.parameters():
            l.append(parameter.requires_grad)
    # print(l)
    return all(l)


def unfreeze_model(model):
    for layer in model.children():
        for parameter in layer.parameters():
            parameter.requires_grad = True


class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNMobileNetAFF(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()

        # print("hi from FPNMobileNetAFF")
        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)
        # print("init of the fpn mobilenetv2 finished")

    def freeze(self):
        self.fpn.freeze()

    def unfreeze(self):
        self.fpn.unfreeze()

    def is_frozen(self):
        l = []
        for param in self.fpn.features.parameters():
            l.append(param.requires_grad)
        return not any(l)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fpn.check_image_size(x)

        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res[:, :, :H, :W], min=0, max=1)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        net = MobileNetV2(n_class=1000)

        if pretrained:
            # Load weights into the project directory
            # state_dict = torch.load(
            #     '/home/mpetru/image_enhancement_model_compression/weights/mobilenet_v2.pth.tar')  # add map_location='cpu' if no gpu
            #
            # net.load_state_dict(state_dict)
            net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.AFFs = nn.ModuleList([
            AFF(160+64+32+24, num_filters),
            AFF(160+64+32+24, num_filters),
            AFF(160+64+32+24, num_filters),
        ])

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)


        self.lateral3 = nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False)


        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)
        self.padder_size = 2 ** 5

        # for param in self.features.parameters():
        #     param.requires_grad = False
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        # print(x.shape)
        # raise Exception('')

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)  # 256
        enc2 = self.enc2(enc1)  # 512
        enc3 = self.enc3(enc2)  # 1024
        enc4 = self.enc4(enc3)  # 2048
        # print(enc0.shape, enc1.shape, enc2.shape, enc3.shape, enc4.shape)

        # Lateral connections
        lateral4 = self.lateral4(enc4)

        # z12 = F.interpolate(res1, scale_factor=0.5)
        # z21 = F.interpolate(res2, scale_factor=2)
        # z42 = F.interpolate(z, scale_factor=2)
        # z41 = F.interpolate(z42, scale_factor=2)

        # print(enc1.shape)
        # print(enc2.shape)
        # print(enc3.shape)
        # print(enc4.shape)
        # torch.Size([1, 24, 64, 64])
        # torch.Size([1, 32, 32, 32])
        # torch.Size([1, 64, 16, 16])
        # torch.Size([1, 160, 8, 8])

        # 16x16
        z34 = F.interpolate(enc4, scale_factor=2)
        z32 = F.interpolate(enc2, scale_factor=0.5)
        z31 = F.interpolate(enc1, scale_factor=0.25)
        # print(z31.shape, z32.shape, enc3.shape, z34.shape)
        # raise Exception('')
        lateral3 = self.AFFs[0](z31, z32, enc3, z34)

        # 32x32
        z34 = F.interpolate(z34, scale_factor=2)
        z33 = F.interpolate(enc3, scale_factor=2)
        z31 = F.interpolate(enc1, scale_factor=0.5)
        lateral2 = self.AFFs[1](z31, enc2, z33, z34)

        # 64x64
        z34 = F.interpolate(z34, scale_factor=2)
        z33 = F.interpolate(z33, scale_factor=2)
        z32 = F.interpolate(enc2, scale_factor=2)
        lateral1 = self.AFFs[2](enc1, z32, z33, z34)

        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4

        map3 = self.td3(self.lateral3(torch.cat([lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest")])))
        map2 = self.td2(self.lateral2(torch.cat([lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest")])))
        map1 = self.td1(self.lateral1(torch.cat([lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest")])))

        # map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        # map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4


from functools import partial
from torch.nn import BatchNorm2d, InstanceNorm2d


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = partial(BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class MIMOFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_layer = get_norm_layer(norm_type="instance")
        self._model = FPNMobileNetAFF(norm_layer=self.norm_layer)
    def forward(self, x):
        return self._model(x)

if __name__ == "__main__":
    model = FPNMobileNetAFF(norm_layer=get_norm_layer(norm_type="instance")).cuda()
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    # model = model.cuda()

    timings = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i in range(100):
        input_img = torch.randn((1, 3, 777, 244)).cuda()
        torch.cuda.synchronize()

        starter.record()

        output = model(input_img)
        # print(output.shape)

        ender.record()

        torch.cuda.synchronize()

        inference_time = starter.elapsed_time(ender)
        print(f"{inference_time=}")
        timings.append(inference_time)

    timings = np.array(timings)
    mean_syn = np.sum(timings) / 100
    std_syn = np.std(timings)

    print(f"seconds average inference time: {mean_syn} +- {std_syn} ms")
