"""This module contains plotting experiments for faithfulness."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
from typing import Tuple, List, Union
import numpy as np
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader


class LeNet(torch.nn.Module):
    """
    LeNet model for MNIST and fMNIST.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True
        )
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torchvision.transforms as transforms


def load_mnist_samples(n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST samples.

    Args:
        n: Number of samples to load.

    Returns:
        Tuple of images and labels.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=n, shuffle=True)
    return next(iter(test_loader))


def load_fmnist_samples(n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load FashionMNIST samples.

    Args:
        n: Number of samples to load.

    Returns:
        Tuple of images and labels.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=n, shuffle=True)
    return next(iter(test_loader))


def modify_and_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    img_size: int = 28,
    modification: str = "black",
) -> np.ndarray:
    """
    Modify images and predict using the model.

    Args:
        model: The model to use for prediction.
        images: Input images.
        labels: Ground truth labels.
        img_size: Size of the images.
        modification: Type of modification to apply.

    Returns:
        Array of confidences.
    """
    num_pixels = img_size * img_size
    confidences = np.zeros((images.shape[0], num_pixels))
    pixel_indices = torch.randperm(num_pixels)
    modified_images = images.clone()
    mean = images.mean()

    for i in range(num_pixels):
        pixel_index = pixel_indices[i]
        flat_images = modified_images.view(images.shape[0], -1)

        if modification == "black":
            flat_images[:, pixel_index] = 0
        elif modification == "white":
            flat_images[:, pixel_index] = 1
        elif modification == "random":
            flat_images[:, pixel_index] = torch.rand(images.shape[0])
        elif modification == "mean":
            flat_images[:, pixel_index] = mean
        modified_images = flat_images.view_as(images)
        outputs = model(modified_images)

        confidences[:, i] = (
            outputs.gather(1, labels.view(-1, 1)).squeeze().detach().cpu().numpy()
        )

    return confidences


def modify_and_predict_with_patches(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: int,
    img_size: int = 28,
    nr_channels: int = 2,
    modification: str = "black",
) -> np.ndarray:
    """
    Modify image with patches and predict using the model.

    Args:
        model: The model to use for prediction.
        image: Input image.
        label: Ground truth label.
        patch_size: Size of each patch.
        img_size: Size of the image.
        nr_channels: Number of channels in the image.
        modification: Type of modification to apply.

    Returns:
        Array of confidences.
    """
    assert modification == "black"
    num_pixels = img_size * img_size
    confidences = np.zeros(num_pixels)
    modified_image = image.clone()
    patches_per_row = img_size // patch_size
    total_patches = patches_per_row**nr_channels
    patch_indices = [
        (i, j)
        for i in range(0, img_size, patch_size)
        for j in range(0, img_size, patch_size)
    ]
    np.random.shuffle(patch_indices)
    patch_count = 0
    pixel_count = 0

    for x, y in patch_indices:
        modified_image[0, 0, x : x + patch_size, y : y + patch_size] = 0
        patch_count += 1
        pixel_count += patch_size**2
        with torch.no_grad():
            outputs = model(modified_image)
            confidence = outputs.gather(1, label.view(-1, 1)).squeeze().item()
            confidences[pixel_count - patch_size**2 : pixel_count] = confidence
            if pixel_count >= num_pixels:
                break

    return confidences
