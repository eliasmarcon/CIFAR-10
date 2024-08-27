from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple



class Subset(Enum):
    
    '''
    Dataset subsets.
    '''

    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class Dataset(ABC):
    
    '''
    Base class of all datasets.
    '''

    @abstractmethod
    def __len__(self) -> int:
        
        '''
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        '''
        
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple: The sample at the given index.
        '''
        
        pass


class ClassificationDataset(Dataset):
    
    '''
    Base class of image classification datasets.
    Sample data are numpy arrays of shape (rows, cols) (grayscale) or (rows, cols, channels) (color).
    Sample labels are integers from 0 to num_classes() - 1.
    '''

    @abstractmethod
    def num_classes(self) -> int:
        
        '''
        Returns the number of classes.

        Returns:
            int: Number of classes.
        '''
        
        pass
