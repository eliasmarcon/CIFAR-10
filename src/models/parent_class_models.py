import torch
import os
from pathlib import Path
from typing import Optional



class BaseModel(torch.nn.Module):

    """
    Base class for PyTorch models providing methods to save and load model weights.
    """

    def save(self, save_dir: Path, suffix: Optional[str] = None) -> None:

        """
        Saves the model's state dictionary to a file.

        Args:
            save_dir (Path): Directory where the model will be saved.
            suffix (Optional[str]): Optional suffix for the filename. If provided, it will be added to the filename.
        """

        filename = os.path.join(save_dir, f"{suffix}.pt" if suffix else "model.pt")
        torch.save(self.state_dict(), filename)


    def load(self, path: Path) -> None:

        """
        Loads the model's state dictionary from a file.

        Args:
            path (Path): Path to the file containing the saved state dictionary.
        """

        self.load_state_dict(torch.load(path))
