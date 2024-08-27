import torchvision
import torchvision.transforms.v2
import torch

from typing import Tuple, Optional
from enum import Enum

# own modules
from dataset.cifar10 import CIFAR10Dataset
from dataset.dataset import Subset



def create_transformations(data_augmentation: bool) -> Tuple[Optional[torchvision.transforms.Compose], Optional[torchvision.transforms.Compose], Optional[torchvision.transforms.Compose]]:
    
    '''
    Creates and returns the transformations for training, validation, and testing datasets.

    Args:
        data_augmentation (bool): Whether to apply data augmentation techniques for the training dataset.

    Returns:
        Tuple[Optional[torchvision.transforms.Compose], Optional[torchvision.transforms.Compose], Optional[torchvision.transforms.Compose]]: Transformations for training, validation, and test datasets.
    '''

    train_transform = None
    val_transform = None
    
    val_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(),
                                                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                    torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    if data_augmentation:
    
        train_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(),
                                                          torchvision.transforms.RandomHorizontalFlip(p=0.5), # added horizontal flip
                                                          torchvision.transforms.RandomRotation(degrees=15), # added random rotation
                                                          torchvision.transforms.RandomCrop(size=(32, 32), padding=4), # added random crop
                                                          torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                          torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    else:
        
        train_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(),
                                                          torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                          torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])


    return train_transform, val_transform, val_transform


def create_dataloaders(dataset_type: Enum, dataset: CIFAR10Dataset, batch_size: int) -> torch.utils.data.DataLoader:
    
    '''
    Creates and returns a DataLoader for the specified dataset type.

    Args:
        dataset_type (Enum): The subset type, either TRAINING, VALIDATION, or TEST.
        dataset (CIFAR10Dataset): The dataset to load.
        batch_size (int): The batch size for loading data.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the specified dataset.
    '''
    
    if dataset_type == Subset.TRAINING:

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=6,
                                           drop_last=False,
                                           persistent_workers=True
                                        )
    
    elif dataset_type == Subset.VALIDATION:
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=6,
                                           drop_last=False,
                                           persistent_workers=True                                                    
                                        )
        
    elif dataset_type == Subset.TEST:
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=False
                                        )
    

def get_datasets(base_path: str = "./cifar-10", data_augmentation: bool = False) -> Tuple[CIFAR10Dataset, CIFAR10Dataset, CIFAR10Dataset]:
    
    '''
    Loads and returns the CIFAR-10 datasets for training, validation, and testing.

    Args:
        base_path (str): The directory containing the CIFAR-10 dataset files. Defaults to "./cifar-10".
        data_augmentation (bool): Whether to apply data augmentation to the training dataset. Defaults to False.

    Returns:
        Tuple[CIFAR10Dataset, CIFAR10Dataset, CIFAR10Dataset]: The training, validation, and test datasets.
    '''

    train_transform, val_transform, test_transform = create_transformations(data_augmentation)
    
    train_data = CIFAR10Dataset(base_path, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(base_path, Subset.VALIDATION, transform=val_transform)
    test_data = CIFAR10Dataset(base_path, Subset.TEST, transform=test_transform)
    
    return train_data, val_data, test_data


def get_dataloaders(train_data: CIFAR10Dataset, val_data: CIFAR10Dataset, test_data: CIFAR10Dataset, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    '''
    Creates and returns the DataLoaders for the training, validation, and testing datasets.

    Args:
        train_data (CIFAR10Dataset): The training dataset.
        val_data (CIFAR10Dataset): The validation dataset.
        test_data (CIFAR10Dataset): The test dataset.
        batch_size (int): The batch size for loading data.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The DataLoaders for the training, validation, and testing datasets.
    '''

    train_loader = create_dataloaders(Subset.TRAINING, train_data, batch_size)
    val_loader = create_dataloaders(Subset.VALIDATION, val_data, batch_size)
    test_loader = create_dataloaders(Subset.TEST, test_data, batch_size)
    
    return train_loader, val_loader, test_loader
    
    
    