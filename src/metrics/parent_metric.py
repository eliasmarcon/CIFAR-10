import torch
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for metrics.

    This class defines the structure for any performance metric used during model evaluation.
    Any metric inheriting from this class must implement the following methods:
    - `reset`: Resets the metric's internal state.
    - `update`: Updates the metric state based on model predictions and ground-truth targets.
    - `__str__`: Returns a string representation of the current metric performance.
    """


    @abstractmethod
    def reset(self) -> None:
        
        """
        Resets the internal state of the metric.

        This method is intended to be called before evaluating a new dataset or epoch.
        It should clear any stored values or counters used for calculating the metric.
        """
        
        pass


    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        
        """
        Updates the metric state based on the model's predictions and the ground-truth targets.

        Args:
            prediction (torch.Tensor): The predicted values from the model. Expected shape is (batch_size, num_classes).
            target (torch.Tensor): The true class labels. Expected shape is (batch_size,).

        This method will be called repeatedly during evaluation to accumulate the metric values.
        """
        
        pass


    @abstractmethod
    def __str__(self) -> str:
        
        """
        Returns a string representation of the current metric performance.

        This is useful for logging and reporting the metric's results after an evaluation is complete.

        Returns:
            str: A string detailing the current metric score, e.g., "Accuracy: 85.5%".
        """
        
        pass