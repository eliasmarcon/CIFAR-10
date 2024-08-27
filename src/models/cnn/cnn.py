import torch
import torch.nn as nn

# own modules
from models.parent_class_models import BaseModel



class CNNBasic(BaseModel):
    
    """
    A basic convolutional neural network (CNN) for image classification, 
    designed for datasets like CIFAR-10.

    Args:
        num_classes (int): The number of output classes for classification.
    """

    def __init__(self, num_classes: int) -> None:
        
        """
        Initializes the CNNBasic model with convolutional, batch normalization, 
        pooling, and fully connected layers.

        Args:
            num_classes (int): Number of output classes (e.g., 10 for CIFAR-10).
        """
        
        super(CNNBasic, self).__init__()
                        
        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(192)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.dense_1 = nn.Linear(512 * 4 * 4, 1024)  # Adjust the input size based on image size and pooling
        self.dense_2 = nn.Linear(1024, 512)
        self.dense_3 = nn.Linear(512, 256)
        self.dense_4 = nn.Linear(256, num_classes)  # Number of classes in CIFAR-10
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.15)
        
        # Softmax layer to convert logits to probabilities
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) representing class probabilities.
        """
        
        # Convolutional Block 1
        x = torch.nn.functional.gelu(self.conv1_1_bn(self.conv1_1(x)))
        x = torch.nn.functional.gelu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(x)

        # Convolutional Block 2
        x = torch.nn.functional.gelu(self.conv2_1_bn(self.conv2_1(x)))
        x = torch.nn.functional.gelu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(x)

        # Convolutional Block 3
        x = torch.nn.functional.gelu(self.conv3_1_bn(self.conv3_1(x)))
        x = torch.nn.functional.gelu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Apply dropout
        x = self.dropout(x)

        # Flatten the feature maps for fully connected layers
        x = self.flatten(x)

        # Fully connected layers with dropout
        x = torch.nn.functional.gelu(self.dense_1(x))
        x = self.dropout(x)
        x = torch.nn.functional.gelu(self.dense_2(x))
        x = self.dropout(x)
        x = torch.nn.functional.gelu(self.dense_3(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.dense_4(x)
        
        # Apply softmax for class probabilities
        x = self.softmax(x)
        
        return x
