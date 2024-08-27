import torch
from typing import List

# own modules
from metrics.parent_metric import Metric



class Accuracy(Metric):
    
    """
    Average classification accuracy.

    Attributes:
        classes (List[str]): List of class names.
        total_predictions (int): Total number of predictions made.
        correct_predictions (int): Number of correct predictions.
        correct_per_class (List[int]): List tracking correct predictions per class.
        total_per_class (List[int]): List tracking total predictions per class.
    """


    def __init__(self, classes: List[str]) -> None:
        
        """
        Initializes the Accuracy metric.

        Args:
            classes (List[str]): List of class names for tracking per-class accuracy.
        """
        
        self.classes = classes
        self.reset()


    def reset(self) -> None:
        
        """
        Resets the internal state of the metric.
        """
        
        self.total_predictions = 0
        self.correct_predictions = 0
        
        # Initialize lists to track correct and total predictions per class
        self.correct_per_class = [0] * len(self.classes)
        self.total_per_class = [0] * len(self.classes)


    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        
        """
        Update the metric by comparing predictions with ground-truth targets.

        Args:
            prediction (torch.Tensor): Predicted scores with shape (s, c), where each row is a class-score vector.
            target (torch.Tensor): Ground-truth labels with shape (s,), containing values between 0 and c-1.

        Raises:
            ValueError: If the data shapes or values are unsupported.
        """
        
        # Get the predicted class for each instance
        predicted_classes = torch.argmax(prediction, dim=1)

        # Validate prediction and target shapes
        if prediction.shape[0] != target.shape[0]:
            raise ValueError('Data shape is unsupported')
        
        if prediction.shape[1] != len(self.classes):
            raise ValueError('Data shape is unsupported')
        
        if not torch.all(target >= 0) or not torch.all(target < len(self.classes)):
            raise ValueError('Data values are unsupported')
        
        # Update the counters
        for i in range(target.shape[0]):
            self.total_predictions += 1
            self.total_per_class[target[i]] += 1
            
            # Check if the prediction matches the target
            if predicted_classes[i] == target[i]:
                self.correct_predictions += 1
                self.correct_per_class[target[i]] += 1


    def __str__(self) -> str:
        
        """
        Returns a string representation of the performance, including overall and per-class accuracy.

        Returns:
            str: A string representation of the metric results.
        """
        
        overall_accuracy = self.accuracy()
        per_class_accuracy = self.per_class_accuracy()
        
        # Create a representation string
        performance_str = f"Overall Accuracy: {overall_accuracy:.4f}\n"
        
        # Calculate the mean per-class accuracy
        performance_str += f"Per-Class Accuracy: {per_class_accuracy:.4f}\n"
        
        # Get the maximum class name length
        max_class_name_length = max([len(class_label) for class_label in self.classes])
        
        for idx, class_label in enumerate(self.classes):
            class_accuracy = self.correct_per_class[idx] / self.total_per_class[idx] if self.total_per_class[idx] > 0 else 0.0
            performance_str += f"   - Accuracy Class {class_label:<{max_class_name_length}}: {class_accuracy:.4f}\n"
        
        return performance_str


    def accuracy(self) -> float:
        
        """
        Computes and returns the overall accuracy as a float between 0 and 1.

        Returns:
            float: The overall accuracy. Returns 0 if no data is available.
        """
        
        if self.total_predictions == 0:
            return 0.0
        
        return self.correct_predictions / self.total_predictions


    def per_class_accuracy(self) -> float:
        
        """
        Computes and returns the mean per-class accuracy as a float between 0 and 1.

        Returns:
            float: The mean per-class accuracy. Returns 0 if no data is available.
        """
        
        per_class_accs = [
            (self.correct_per_class[i] / self.total_per_class[i]) if self.total_per_class[i] > 0 else 0.0
            for i in range(len(self.classes))
        ]
        
        if sum(per_class_accs) == 0:
            return 0.0
        else:
            return sum(per_class_accs) / len(self.classes)


    def get_per_class_accuracy(self) -> List[float]:
        
        """
        Returns a list of per-class accuracies.

        Returns:
            List[float]: List of accuracies for each class.
        """
        
        return [
            (self.correct_per_class[i] / self.total_per_class[i]) if self.total_per_class[i] > 0 else 0.0
            for i in range(len(self.classes))
        ]