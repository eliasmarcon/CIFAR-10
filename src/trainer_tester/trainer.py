import torch
import logging
import time

from typing import Tuple, Optional



class Trainer:
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 train_loader: torch.utils.data.DataLoader,
                 train_metric,
                 val_loader: torch.utils.data.DataLoader,
                 val_metric,
                 device: torch.device,
                 wandb_logger: Optional[object],
                 save_dir: str,
                 batch_size: int,
                 val_freq: int,
                 early_stopping_patience: int) -> None:
        
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            loss_fn (torch.nn.Module): The loss function.
            lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The learning rate scheduler.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            train_metric: Metric to evaluate the training performance.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            val_metric: Metric to evaluate the validation performance.
            device (torch.device): Device to run the training on.
            wandb_logger (Optional[object]): Logger to record the results, can be None.
            save_dir (str): Directory to save model checkpoints.
            batch_size (int): Size of the training batches.
            val_freq (int): Frequency (in epochs) to perform validation.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.train_metric = train_metric
        self.val_loader = val_loader
        self.val_metric = val_metric
        self.device = device
        self.wandb_logger = wandb_logger
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize other variables
        self.num_train_data = len(train_loader.dataset)
        self.num_val_data = len(val_loader.dataset)
        self.best_val_loss = float('inf')
        self.best_acc = 0.0
        self.best_pcacc = 0.0


    def train(self, epochs: int) -> None:
        
        """
        Train the model for the specified number of epochs, using the `_train_epoch` and `_val_epoch` methods.
        Saves the model if the mean per-class accuracy on the validation dataset is higher than the currently saved best.
        Validation is performed based on `val_freq`, and early stopping is applied if validation performance does not improve.

        Args:
            epochs (int): Number of epochs to train the model.

        Returns:
            None
        """
        
        patience_counter = 0
        
        for epoch_idx in range(1, epochs + 1):
            starting_time = time.time()

            # Train for one epoch
            train_loss, train_acc, train_pcacc = self._train_epoch()

            # Perform validation if needed
            if epoch_idx % self.val_freq == 0:
                val_loss, val_acc, val_pcacc = self._val_epoch()

                # Check and save the best model based on accuracy
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.model.save(self.save_dir, f"best_val_acc")

                # Check and save the best model based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0  # Reset patience counter
                    self.model.save(self.save_dir, f"best_val_loss")
                
                else:
                    patience_counter += 1  # Increment patience counter
                    
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered. Training stopped after {epoch_idx} epochs.")
                    break
            
            # Step the learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            # Log the current metrics
            logging.info(
                f"Epoch {epoch_idx:2}/{epochs} completed in {(time.time() - starting_time):4.2f} seconds | "
                f"Train Loss: {train_loss:6.4f} | Train Accuracy: {train_acc:6.4f} | Train per Class Accuracy: {train_pcacc:6.4f} | "
                f"Val Loss: {val_loss:6.4f} | Val Accuracy: {val_acc:6.4f} | Val per Class Accuracy: {val_pcacc:6.4f}"
            )
               
            if self.wandb_logger:
                
                self.wandb_logger.log({
                    f"train_loss": train_loss,
                    f"train_accuracy": train_acc,
                    f"train_per_class_accuracy": train_pcacc,
                    f"val_loss": val_loss if (epoch_idx % self.val_freq == 0) else None,
                    f"val_accuracy": val_acc if (epoch_idx % self.val_freq == 0) else None,
                    f"val_per_class_accuracy": val_pcacc if (epoch_idx % self.val_freq == 0) else None,
                })
                
                # Log per-class accuracy
                per_class_accs_train = self.train_metric.get_per_class_accuracy()
                per_class_accs_val = self.val_metric.get_per_class_accuracy()
                class_names_val = self.val_metric.classes
                
                for per_class_acc_train, per_class_acc_val, class_name in zip(per_class_accs_train, per_class_accs_val, class_names_val):
                    self.wandb_logger.log({
                        f"train_per_class_accuracy_{class_name.lower()}": per_class_acc_train,
                        f"val_per_class_accuracy_{class_name.lower()}": per_class_acc_val
                    })
                            
        # Save the final model
        self.model.save(self.save_dir, f"terminal")
    
    
    def _train_epoch(self) -> Tuple[float, float, float]:
        
        """
        Training logic for one epoch. Resets metrics, performs forward and backward passes, and updates the model parameters.

        Returns:
            Tuple[float, float, float]: Average loss, mean accuracy, and mean per-class accuracy for the epoch.
        """
        
        self.train_metric.reset()
        epoch_loss = 0.0
        
        # Set the model to training mode
        self.model.train()
        
        for inputs, targets in self.train_loader:
            
            inputs, targets = inputs.to(self.device), targets.to(self.device).long()
            batch_size = inputs.shape[0]
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size
            self.train_metric.update(outputs, targets)
                    
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss /= self.num_train_data
        acc = self.train_metric.accuracy()
        pcacc = self.train_metric.per_class_accuracy()
        
        return epoch_loss, acc, pcacc
    
    
    def _val_epoch(self) -> Tuple[float, float, float]:
        
        """
        Validation logic for one epoch. Computes loss and metrics without updating the model parameters.

        Returns:
            Tuple[float, float, float]: Average loss, mean accuracy, and mean per-class accuracy for the epoch.
        """
        
        self.val_metric.reset()
        epoch_loss = 0.0
        
        self.model.eval()
        
        with torch.no_grad():
            
            for inputs, targets in self.val_loader:
                
                # Get the inputs and targets, and move them to the device
                inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                batch_size = inputs.shape[0]
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Update the loss
                epoch_loss += loss.item() * batch_size
                
                # Update the validation metric
                self.val_metric.update(outputs, targets)

        # Calculate average loss for the validation set
        epoch_loss /= self.num_val_data
        acc = self.val_metric.accuracy()
        pcacc = self.val_metric.per_class_accuracy()
        
        return epoch_loss, acc, pcacc
