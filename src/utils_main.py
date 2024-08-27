import torch
from pathlib import Path


def create_save_dir(model_type: str, model_run_name: str, model_save_dir: Path = Path("saved_models")) -> Path:
    
    """
    Create a directory for saving the model checkpoints.

    Args:
        model_type (str): The type of model (e.g., 'resnet', 'cnn', 'vit').
        model_run_name (str): The specific run name or identifier for this model run.
        model_save_dir (Path, optional): The base directory for saving models. Defaults to "saved_models".

    Returns:
        Path: The full path to the directory for saving this model run.
    """
    
    # Create the base save directory if it does not exist
    save_dir = model_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the model type directory if it does not exist
    model_type_save_dir = save_dir / model_type
    model_type_save_dir.mkdir(exist_ok=True)
    
    # Create the model run name directory if it does not exist
    model_run_name_save_dir = model_type_save_dir / model_run_name
    model_run_name_save_dir.mkdir(exist_ok=True)
    
    return model_run_name_save_dir


def create_lr_scheduler(scheduler : str, gamma : float, optimizer : torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    
    """
    Create a learning rate scheduler.
    """
    
    # Create a learning rate scheduler
    if scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=gamma)
        
    elif scheduler == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    else:
        raise NotImplementedError(f"Scheduler {scheduler} is not implemented.")
    
    return lr_scheduler
