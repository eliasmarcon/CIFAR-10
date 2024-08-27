import torch
import torch.nn as nn
from typing import List

# own modules
from models.parent_class_models import BaseModel



class ResNet(BaseModel):
    
    """
    Implements the ResNet model with configurable layers and block structure.

    Args:
        block (nn.Module): The block type to be used (e.g., `ResidualBlock`).
        layers (List[int]): Number of blocks in each of the four layers.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(self, block: nn.Module, layers: List[int], num_classes: int = 10) -> None:
        
        """
        Initializes the ResNet model.

        Args:
            block (nn.Module): The block type (e.g., `ResidualBlock`).
            layers (List[int]): A list containing the number of blocks for each layer.
            num_classes (int, optional): Number of output classes. Default is 10.
        """
        
        super(ResNet, self).__init__()

        self.inplanes = 64

        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        
        """
        Creates a sequential layer consisting of multiple residual blocks.

        Args:
            block (nn.Module): The block type to use.
            planes (int): Number of output channels for this layer.
            blocks (int): Number of blocks in this layer.
            stride (int, optional): Stride value for the first block. Default is 1.

        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
        
        downsample = None

        # Downsample if stride is not 1 or inplanes do not match planes
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Defines the forward pass for the ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    
    """
    Implements a residual block used in the ResNet architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the first convolution. Default is 1.
        downsample (Optional[nn.Module], optional): Downsampling layer if needed. Default is None.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None) -> None:
        
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the first convolution. Default is 1.
            downsample (Optional[nn.Module], optional): Downsampling layer. Default is None.
        """
        
        super(ResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Defines the forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).
        """
        
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
