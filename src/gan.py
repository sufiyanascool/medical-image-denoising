import torch
import torch.nn as nn


def down_block(in_channels, out_channels, use_batch_norm=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                  bias=not use_batch_norm)
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                           bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetGenerator(nn.Module):
    """U-Net generator for paired grayscale image translation at 256 x 256."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = down_block(in_channels, 64, use_batch_norm=False)  # 128 x 128
        self.down2 = down_block(64, 128)                               # 64 x 64
        self.down3 = down_block(128, 256)                              # 32 x 32
        self.down4 = down_block(256, 512)                              # 16 x 16
        self.down5 = down_block(512, 512)                              # 8 x 8
        self.down6 = down_block(512, 512)                              # 4 x 4
        self.down7 = down_block(512, 512)                              # 2 x 2

        self.up1 = up_block(512, 512)
        self.up2 = up_block(1024, 512)
        self.up3 = up_block(1024, 512)
        self.up4 = up_block(1024, 256)
        self.up5 = up_block(512, 128)
        self.up6 = up_block(256, 64)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        d1 = self.down1(image)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = torch.cat((self.up1(d7), d6), dim=1)
        u2 = torch.cat((self.up2(u1), d5), dim=1)
        u3 = torch.cat((self.up3(u2), d4), dim=1)
        u4 = torch.cat((self.up4(u3), d3), dim=1)
        u5 = torch.cat((self.up5(u4), d2), dim=1)
        u6 = torch.cat((self.up6(u5), d1), dim=1)
        return self.output_layer(u6)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator; input is concatenated noisy/target images."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, noisy_image, target_image):
        pair = torch.cat((noisy_image, target_image), dim=1)
        return self.network(pair)
