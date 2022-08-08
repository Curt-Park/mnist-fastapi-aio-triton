"""Functions for data handling.

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""

from typing import Any, Dict, Tuple

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

MEAN = 0.1307
STD = 0.3081


def get_preprocessor_train() -> transforms.Compose:
    """Get a preprocessor for mnist."""
    transform_seq = [
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ]
    transform = transforms.Compose(transform_seq)
    return transform


def get_preprocessor_test() -> transforms.Compose:
    """Get a preprocessor for test for mnist."""
    transform_seq = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    transform = transforms.Compose(transform_seq)
    return transform


def get_dataloaders(
    batch_size: int, test_batch_size: int, use_cuda: bool
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for training and test."""
    # set kwargs for training and test
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs: Dict[str, Any] = {"batch_size": batch_size}
    test_kwargs: Dict[str, Any] = {"batch_size": test_batch_size}
    train_kwargs.update(cuda_kwargs if use_cuda else {})
    test_kwargs.update(cuda_kwargs if use_cuda else {})

    # set dataset loaders
    transform_train = get_preprocessor_train()
    transform_test = get_preprocessor_test()
    data_path = "../../data"
    dataset1 = datasets.MNIST(
        data_path, train=True, download=True, transform=transform_train
    )
    dataset2 = datasets.MNIST(data_path, train=False, transform=transform_test)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader
