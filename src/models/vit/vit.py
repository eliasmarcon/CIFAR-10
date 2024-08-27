import torch
from torch import nn
from pathlib import Path
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple

# own modules
from models.parent_class_models import BaseModel

# SOURCE: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py



# Helper Functions
def pair(t: int) -> Tuple[int, int]:
    
    """
    Converts a single integer into a tuple with the same value for both elements.

    Args:
        t (int): The value to convert.

    Returns:
        Tuple[int, int]: A tuple containing (t, t).
    """
    
    return (t, t) if isinstance(t, int) else t


def posemb_sincos_2d(h: int, w: int, dim: int, temperature: int = 10000, dtype=torch.float32) -> torch.Tensor:
    
    """
    Generates 2D sine-cosine positional embeddings.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        dim (int): Dimensionality of the embeddings.
        temperature (int, optional): Scaling factor for the positional embeddings. Default is 10000.
        dtype (torch.dtype, optional): Data type of the embeddings. Default is torch.float32.

    Returns:
        torch.Tensor: A tensor containing the positional embeddings.
    """
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "Feature dimension must be a multiple of 4 for sine-cosine embeddings."
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    
    return pe.type(dtype)


class FeedForward(nn.Module):
    
    """
    FeedForward network used in the Vision Transformer (ViT).

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the FeedForward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        
        return self.net(x)


class Attention(nn.Module):
    
    """
    Multi-head self-attention mechanism used in the Vision Transformer (ViT).

    Args:
        dim (int): Input dimension.
        heads (int, optional): Number of attention heads. Default is 8.
        dim_head (int, optional): Dimension of each attention head. Default is 64.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        
        super().__init__()
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class Transformer(nn.Module):
    
    """
    Transformer encoder block consisting of multiple attention and feedforward layers.

    Args:
        dim (int): Input dimension.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the feedforward network.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]) for _ in range(depth)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return self.norm(x)


class ViT(BaseModel):
    
    """
    Vision Transformer (ViT) model for image classification.

    Args:
        image_size (int or Tuple[int, int]): Size of the input image.
        patch_size (int or Tuple[int, int]): Size of each patch.
        num_classes (int): Number of output classes.
        dim (int): Dimension of the embedding space.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the feedforward network.
        channels (int, optional): Number of input channels. Default is 3 (RGB images).
        dim_head (int, optional): Dimension of each attention head. Default is 64.
    """

    def __init__(self, *, image_size: int, patch_size: int, num_classes: int, dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        
        super(ViT, self).__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)


    def forward(self, img: torch.Tensor) -> torch.Tensor:

        """
        Forward pass through the Vision Transformer.

        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        
        return self.linear_head(x)
