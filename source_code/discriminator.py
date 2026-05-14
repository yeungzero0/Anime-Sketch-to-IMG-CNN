import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    stride,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_sizes=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels * 2,
                    feature_sizes[0],
                    4,
                    2,
                    1,
                    padding_mode="reflect",
                )
            ),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = feature_sizes[0]
        for feature in feature_sizes[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == feature_sizes[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                )
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, input_image, target_image, return_features=False):
        combined_input = torch.cat([input_image, target_image], dim=1)
        x = self.initial(combined_input)
        features = [x]
        for layer in self.model:
            x = layer(x)
            features.append(x)
        if return_features:
            return x, features[:-1]  # Exclude final output from features
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, scales=[1.0, 0.5]):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(in_channels=in_channels) for _ in range(len(scales))
        ])
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.scales = scales

    def forward(self, input_image, target_image, return_features=False):
        outputs = []
        features = []
        curr_input, curr_target = input_image, target_image
        for i, disc in enumerate(self.discriminators):
            if return_features:
                out, feats = disc(curr_input, curr_target, return_features=True)
                outputs.append(out)
                features.append(feats)
            else:
                outputs.append(disc(curr_input, curr_target))
            if i < len(self.discriminators) - 1:  # Downsample for next scale
                curr_input = self.downsample(curr_input)
                curr_target = self.downsample(curr_target)
        if return_features:
            return outputs, features
        return outputs