import pickle
import numpy as np
import os

from typing import Tuple, Optional

# own modules
from dataset.dataset import Subset, ClassificationDataset



class CIFAR10Dataset(ClassificationDataset):
    
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform: Optional[callable] = None) -> None:
        
        '''
        Loads the dataset from a directory `fdir` that contains the Python version
        of the CIFAR-10, i.e., files "data_batch_1", "test_batch" and so on.
        Raises ValueError if `fdir` is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.

        Args:
            fdir (str): Directory containing the CIFAR-10 dataset files.
            subset (Subset): The subset to load, one of TRAINING, VALIDATION, or TEST.
            transform (callable, optional): A function/transform to apply to the images.
        '''

        self.classes = ("Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck")
        
        # Set the transform
        self.transform = transform
        
        self.data = None
        self.labels = []
        
        # Check if fdir is a directory        
        if not os.path.isdir(fdir):
            raise ValueError('fdir is not a directory')
        
        # Load the training set
        if subset == Subset.TRAINING:
            
            for i in range(1, 5):
                
                if not os.path.isfile(f"{fdir}/data_batch_{str(i)}"):
                    raise ValueError(f'data_batch_{str(i)} is missing')
                
                else:    
                    with open(f"{fdir}/data_batch_{str(i)}", 'rb') as file:
                        
                        dict_data = pickle.load(file, encoding='bytes')
                        
                        self.data = np.vstack((self.data, dict_data[b'data'])) if i > 1 else dict_data[b'data']
                        self.labels += dict_data[b'labels']
            
            # reshape the data to (num_samples, 3, 32, 32)
            self.data, self.labels = self.transform_data_labels(self.data, self.labels)
              
        # Load the validation set
        elif subset == Subset.VALIDATION:
            
            if not os.path.isfile(f"{fdir}/data_batch_5"):
                raise ValueError('data_batch_5 is missing')
            
            else:
                with open(f"{fdir}/data_batch_5", 'rb') as file:

                    dict_data = pickle.load(file, encoding='bytes')
                        
                    self.data = dict_data[b'data']
                    self.labels += dict_data[b'labels']
            
            # reshape the data to (num_samples, 3, 32, 32)
            self.data, self.labels = self.transform_data_labels(self.data, self.labels)

            
        # Load the test set
        elif subset == Subset.TEST:
            
            if not os.path.isfile(f"{fdir}/test_batch"):
                raise ValueError('test_batch is missing')
            
            else:
                with open(f"{fdir}/test_batch", 'rb') as file:
                    
                    dict_data = pickle.load(file, encoding='bytes')
                        
                    self.data = dict_data[b'data']
                    self.labels += dict_data[b'labels']
            
            # reshape the data to (num_samples, 3, 32, 32)
            self.data, self.labels = self.transform_data_labels(self.data, self.labels)
    

    def __len__(self) -> int:
        
        '''
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        '''  
        
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        
        '''
        Returns the `idx`-th sample in the dataset, which is a tuple
        consisting of the image and label.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the image and label.
        '''
        
        if idx >= len(self.data):
            raise IndexError('Index out of bounds')
        
        else:
            image = self.data[idx]
            label = self.labels[idx]
            
            if self.transform is not None:
                image = self.transform(image)
            
            return (image, label)


    def num_classes(self) -> int:
        
        '''
        Returns the number of classes.

        Returns:
            int: Number of unique classes.
        '''
        
        return len(np.unique(self.labels))
    
    
    def transform_data_labels(self, data: np.ndarray, labels: list) -> Tuple[np.ndarray, np.ndarray]:
        
        '''
        Reshapes and reorders the image data and converts labels to a numpy array.

        Args:
            data (np.ndarray): Raw image data of shape (num_samples, 3072).
            labels (list): List of labels corresponding to the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of reshaped image data of shape (num_samples, 32, 32, 3)
                                           and corresponding labels.
        '''

        data = data.reshape((len(data), 3, 32, 32))
        data = np.rollaxis(data, 1, 4)
        labels = np.array(labels)
        
        return data, labels

