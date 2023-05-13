import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable: bool = True) -> int:

    if trainable:
        num_parameters = sum(p.numel() for p in module.parameters()
                             if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in module.parameters())

    return num_parameters


def conv_parameters(in_channels, out_channels, kernel_size, bias) -> int:

    num_parameters = in_channels * out_channels * kernel_size[0] * kernel_size[
        1]

    if bias:
        num_parameters += out_channels

    return num_parameters


def separable_conv_parameters(in_channels, out_channels, kernel_size,
                              bias) -> int:

    num_parameters = in_channels * kernel_size[0] * kernel_size[
        1] + in_channels * out_channels

    if bias:
        num_parameters += (in_channels + out_channels)

    return num_parameters


class DepthwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode,
                                        device=device,
                                        dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)

        return x


class PointwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pointwise_conv(x)

        return x


class SeparableConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode,
                                              device=device,
                                              dtype=dtype)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              bias=bias,
                                              device=device,
                                              dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


if __name__ == "__main__":

    input_size = (128, 128)
    in_channels = 8
    out_channels = 64
    kernel_size = (3, 3)
    bias = True

    conv = nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     bias=bias)
    separable_conv = SeparableConv2D(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     bias=bias)

    num_parameters_conv = count_parameters(module=conv)
    num_parameters_separable_conv = count_parameters(module=separable_conv)

    assert num_parameters_conv == conv_parameters(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  bias=bias)
    assert num_parameters_separable_conv == separable_conv_parameters(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=bias)

    random_input = torch.rand((1, in_channels, *input_size))
    assert conv(random_input).shape == separable_conv(random_input).shape

    print(f"Input Size: {input_size}, In Channels: {in_channels}, "
          f"Out Channels: {out_channels}, Kernel Size: {kernel_size}, "
          f"Bias: {bias}.")
    print(f"Number of Parameters for Conv: {num_parameters_conv}.")
    print(f"Number of Parameters for Separable Conv: "
          f"{num_parameters_separable_conv}.")

    try:
        # pip install ptflops
        from ptflops import get_model_complexity_info
        conv_macs, params = get_model_complexity_info(
            model=conv,
            input_res=(in_channels, *input_size),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False)
        separable_conv_macs, params = get_model_complexity_info(
            model=separable_conv,
            input_res=(in_channels, *input_size),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False)
        print(f"Number of MACs for Conv: {conv_macs}.")
        print(f"Number of MACs for Separable Conv: {separable_conv_macs}.")
    except:
        pass