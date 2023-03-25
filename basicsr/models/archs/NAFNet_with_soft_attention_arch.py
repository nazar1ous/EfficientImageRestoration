# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.NAFNet_arch import NAFBlock, SimpleGate


class Attention_block(nn.Module):
    def __init__(self, g_channel, x_channel, out_channel):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channel, out_channel, kernel_size=1, stride=2, padding=0, bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.out_layer = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        concat_xg = self.relu(g1 + x1)
        psi = self.psi(concat_xg)
        psi_upsampled = nn.functional.interpolate(psi, scale_factor=2, mode="bilinear")
        y = x * psi_upsampled
        return self.norm(self.out_layer(y))

class SoftANAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.soft_attentions = nn.ModuleList()
        for chan_idx in range(len(dec_blk_nums)):
            channel_n = (width) * (2 ** chan_idx)
            # print(channel_n)
            self.soft_attentions.append(
                Attention_block(channel_n * 2, channel_n, channel_n)
            )

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # print(inp.shape)
        inp = self.check_image_size(inp)
        # print(inp.shape)

        x = self.intro(inp)
        # print(x.shape)
        # raise Exception('')
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for soft_att, decoder, up, enc_skip in zip(self.soft_attentions[::-1], self.decoders, self.ups, encs[::-1]):
            # print(f'{x.shape=}')
            # print(f'{enc_skip.shape=}')
            enc_skip = soft_att(x=enc_skip, g=x)
            # print(enc_skip.shape)
            # print(x.shape, enc_skip.shape)

            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class SoftANAFNetLocal(Local_Base, SoftANAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        SoftANAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

# if __name__ == "__main__":
#     c = 64
#     b, h, w = 2, 64, 64

#     x = torch.randn(2, 128, 128, 128)
#     g = torch.randn(2, 64, 64, 64)
#     sa = SoftAttention(c)
#     res = sa(x, g)



if __name__ == '__main__':
    import numpy as np
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = SoftANAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    model = net
    # net = torch.compile(net, mode="reduce-overhead")



    # inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)
    # net = torch.compile(net, mode="reduce-overhead")

    device = torch.device("cuda")
    net.to(device)
    dummy_input = torch.randn(1, 3,256,256, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    model = net
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'Time={mean_syn} +- {std_syn}')