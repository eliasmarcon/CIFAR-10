import torch

# own modules
import overall_config
from models.cnn.cnn import CNNBasic
from models.resnet.resnet import ResNet, ResidualBlock
from models.vit.vit import ViT


def create_model(model_type: str) -> torch.nn.Module:
    
    """
    Creates and returns a model instance based on the specified model type.

    Args:
        model_type (str): The type of model to create. It should contain one of the keywords: "resnet", "cnn", or "vit".

    Returns:
        torch.nn.Module: An instance of the specified model.

    Raises:
        ValueError: If the provided model type is not recognized.
    """
    
    if "resnet" in model_type:
        model = ResNet(
            ResidualBlock,
            [2, 2, 2, 2],
            overall_config.N_CLASSES
        )
        
    elif "cnn" in model_type:
        model = CNNBasic(overall_config.N_CLASSES)
        
    elif "vit" in model_type:
        model = ViT(
            image_size=overall_config.IMAGE_SIZE,
            patch_size=4,
            num_classes=overall_config.N_CLASSES,
            dim=2048,
            depth=8,
            heads=6,
            channels=3,
            mlp_dim=2048
        )
        
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    
    return model
