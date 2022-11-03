import torch
import torch.nn as nn


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS,
                                     padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.Conv3d(8, cfg.CONST.NUM_CLASSES, kernel_size=1),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, image_features):
        # image_features1 = image_features.permute(0, 2, 1, 3).contiguous()

        # 16,8,256,8,8
        gen_volume = image_features.view(-1, 1024, 2, 2, 2)
        # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
        gen_volume = self.layer4(gen_volume)
        gen_volume = self.layer5(gen_volume)
        gen_volume = self.layer6(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_volume = self.layer7(gen_volume)
        raw_feature = gen_volume

        # gen_volume = self.layer12(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
        # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])

        gen_volumes = (torch.squeeze(gen_volume, dim=1))

        # gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        gen_volumes = self.final_layer(gen_volumes)  # new multilabel layer
        # raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_feature, gen_volumes


class ConvMixer(nn.Module):
    def __init__(self, cfg):
        super(ConvMixer, self).__init__()
        self.cfg = cfg

        self.encoder_final = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(0.2)
        )

        self.encode_path = nn.Sequential(
            nn.Conv2d(3, cfg.NETWORK.DIM, kernel_size=cfg.NETWORK.PATCH_SIZE, stride=cfg.NETWORK.PATCH_SIZE),
            nn.GELU(),
            nn.BatchNorm2d(cfg.NETWORK.DIM),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.NETWORK.DIM, cfg.NETWORK.DIM, cfg.NETWORK.KERNEL_SIZE, groups=cfg.NETWORK.DIM,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(cfg.NETWORK.DIM)
                )),
                nn.Conv2d(cfg.NETWORK.DIM, cfg.NETWORK.DIM, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.NETWORK.DIM)
            ) for i in range(cfg.NETWORK.DEPTH)],
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Flatten(),
            # nn.Linear(cfg.NETWORK.DIM, 256*8*8)
        )

    def forward(self, x):

        total_out = []

        for iter_counter in range(x.shape[1]):
            out = self.encode_path(x[:, iter_counter, :, :, :])
            if iter_counter == 0:
                # out = torch.reshape(out, (x.shape[0], 1, cfg.NETWORK.DIM/64, 8, 8))
                out = torch.reshape(out, (x.shape[0], 1, 128, 8, 8))
                total_out = self.encoder_final(torch.cat((out, out), dim=1))
            else:
                total_out = self.encoder_final(torch.cat((total_out, total_out), dim=1))

        return total_out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
