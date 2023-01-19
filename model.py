from imports import *

class ESPCN(nn.Module):
    def __init__(self, scale_factor, in_c=1) -> None:
        super().__init__()

        # conv2d with 64 filters and kernel size 5x5
        self.first = nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=5, padding=5//2)
        # conv2d with 32 filters and kernel size 3x3
        self.second = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=3//2)
        # conv2d with fixed output channel channel_size x upscale_factor**2 and kernel size 3x3
        self.third = nn.Conv2d(in_channels=32, out_channels=in_c*(scale_factor**2), kernel_size=3, padding=3//2)
        # sub-pixel shuffle function
        self.shuffle = nn.PixelShuffle(scale_factor)

        # tanh function
        self.tanh = nn.Tanh()
        # sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.first(x)
        x = self.tanh(x)
        x = self.second(x)
        x = self.tanh(x)

        x = self.third(x)
        x = self.shuffle(x)
        x = self.sigmoid(x)

        return x

def ESPCN_3(**kwargs) -> ESPCN:
    model = ESPCN(scale_factor=3, **kwargs)
    return model
