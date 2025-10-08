import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 sub-groups with 2 basic blocks each (8 blocks total)
        # First sub-group: 64 channels, stride=1
        self.layer1 = self._make_layer(64, 64, stride=1)
        
        # Second sub-group: 128 channels, stride=2
        self.layer2 = self._make_layer(64, 128, stride=2)
        
        # Third sub-group: 256 channels, stride=2
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        # Fourth sub-group: 512 channels, stride=2
        self.layer4 = self._make_layer(256, 512, stride=2)
        
        # Average pooling and final linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        """Create a layer with 2 basic blocks"""
        layers = []

        # First block (may have stride=2 for downsampling)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Second block (always stride=1)
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x






class BasicBlock(nn.Module):
    """Basic Block for ResNet-18"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out
    






class ResNet18_NoBN(nn.Module):
    """ResNet-18 for CIFAR-10 (without BatchNorm layers)"""
    def __init__(self, num_classes=10):
        super(ResNet18_NoBN, self).__init__()
        
        # Initial convolutional layer (no BN)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # 4 sub-groups with 2 basic blocks each (8 blocks total)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Average pooling and final linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        """Create a layer with 2 basic blocks"""
        layers = []
        layers.append(BasicBlock_NoBN(in_channels, out_channels, stride))
        layers.append(BasicBlock_NoBN(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock_NoBN(nn.Module):
    """Basic Block for ResNet-18 without BatchNorm"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_NoBN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        # No BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # No BatchNorm

        # Shortcut connection (no BN)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out






