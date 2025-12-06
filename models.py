import torch
import torch.nn as nn
import torchvision.models as tv

def _to_1ch_first_conv(conv3: nn.Conv2d) -> nn.Conv2d:
    """
    Convert a pretrained 3-channel first conv to 1-channel by averaging RGB weights.
    """
    conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size,
        stride=conv3.stride,
        padding=conv3.padding,
        bias=conv3.bias is not None,
    )
    with torch.no_grad():
        conv1.weight[:] = conv3.weight.mean(dim=1, keepdim=True)
        if conv3.bias is not None:
            conv1.bias[:] = conv3.bias
    return conv1

def build_model(name: str = "resnet18", in_channels: int = 1) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        if in_channels == 1:
            m.conv1 = _to_1ch_first_conv(m.conv1)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if name == "densenet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1)
        if in_channels == 1:
            m.features.conv0 = _to_1ch_first_conv(m.features.conv0)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        return m

    class BasicCNN(nn.Module):
        def __init__(self, in_ch=1):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
            )
            self.head = nn.Linear(128, 1)
        def forward(self, x):
            x = self.feature(x).flatten(1)
            return self.head(x)

    return BasicCNN(in_channels)
