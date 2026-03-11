"""
Data Augmentation for 3D Medical Imaging (SpineNetV2).

Implements medically appropriate augmentations for lumbar spine MRI:
- Geometric transforms (flip, rotation)
- Intensity transforms (brightness, contrast, noise)
- Class-aware oversampling for minority classes

Author: SpineNetV2 Improved Implementation
"""

import torch
import numpy as np
import random
from typing import Tuple


class RandomHorizontalFlip:
    """
    Randomly flip the volume horizontally (left-right).

    Medical justification: Spine anatomy is roughly symmetric.
    Safe for spinal canal stenosis and foraminal narrowing.
    """
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying the flip (default: 0.5)
        """
        self.p = p

    def __call__(self, volume):
        """
        Args:
            volume: Tensor of shape (9, 112, 224) or (C, D, H, W)
        Returns:
            Flipped or original volume
        """
        if random.random() < self.p:
            # Flip along width dimension (last dimension)
            return torch.flip(volume, dims=[-1])
        return volume


class RandomRotation:
    """
    Randomly rotate the volume by a small angle.

    Medical justification: Patient positioning varies slightly.
    Small rotations (±10°) preserve anatomical features.
    """
    def __init__(self, degrees=10):
        """
        Args:
            degrees: Maximum rotation angle in degrees (default: 10)
        """
        self.degrees = degrees

    def __call__(self, volume):
        """
        Args:
            volume: Tensor of shape (9, 112, 224) or (C, D, H, W)
        Returns:
            Rotated volume
        """
        # Random angle in range [-degrees, +degrees]
        angle = random.uniform(-self.degrees, self.degrees)

        # Convert to numpy for rotation
        volume_np = volume.numpy() if isinstance(volume, torch.Tensor) else volume

        # Rotate each slice in the depth dimension
        rotated_slices = []
        for i in range(volume_np.shape[0]):
            slice_2d = volume_np[i]
            # Create rotation matrix
            h, w = slice_2d.shape
            center = (w // 2, h // 2)
            M = self._get_rotation_matrix(center, angle, scale=1.0)
            # Apply rotation
            rotated = self._warp_affine(slice_2d, M, (w, h))
            rotated_slices.append(rotated)

        rotated_volume = np.stack(rotated_slices, axis=0)

        # Convert back to tensor
        return torch.from_numpy(rotated_volume).float()

    @staticmethod
    def _get_rotation_matrix(center, angle, scale):
        """Compute 2D rotation matrix."""
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)

        cx, cy = center
        M = np.array([
            [scale * cos_val, scale * sin_val, (1 - scale * cos_val) * cx - scale * sin_val * cy],
            [-scale * sin_val, scale * cos_val, scale * sin_val * cx + (1 - scale * cos_val) * cy]
        ], dtype=np.float32)
        return M

    @staticmethod
    def _warp_affine(image, M, output_shape):
        """Simple affine transformation (rotation)."""
        from scipy import ndimage
        # Use scipy for rotation
        h, w = image.shape
        cy, cx = h // 2, w // 2
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        rotated = ndimage.rotate(image, -angle, reshape=False, order=1, mode='nearest')
        return rotated


class RandomBrightnessContrast:
    """
    Randomly adjust brightness and contrast.

    Medical justification: MRI intensity varies across scanners and protocols.
    Augmentation improves generalization.
    """
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        """
        Args:
            brightness_limit: Range for brightness adjustment [-limit, +limit]
            contrast_limit: Range for contrast adjustment [-limit, +limit]
            p: Probability of applying the transform
        """
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, volume):
        """
        Args:
            volume: Tensor of shape (9, 112, 224)
        Returns:
            Adjusted volume
        """
        if random.random() < self.p:
            # Random brightness factor
            brightness = random.uniform(-self.brightness_limit, self.brightness_limit)

            # Random contrast factor
            contrast = random.uniform(-self.contrast_limit, self.contrast_limit)
            contrast_factor = 1.0 + contrast

            # Apply: I' = contrast * I + brightness
            volume = volume * contrast_factor + brightness

            # Clip to valid range [0, 1]
            volume = torch.clamp(volume, 0.0, 1.0)

        return volume


class RandomGaussianNoise:
    """
    Add random Gaussian noise to the volume.

    Medical justification: Simulates scanner noise and artifacts.
    Improves model robustness.
    """
    def __init__(self, std_limit=0.05, p=0.3):
        """
        Args:
            std_limit: Maximum standard deviation of noise
            p: Probability of applying noise
        """
        self.std_limit = std_limit
        self.p = p

    def __call__(self, volume):
        """
        Args:
            volume: Tensor of shape (9, 112, 224)
        Returns:
            Noisy volume
        """
        if random.random() < self.p:
            # Random noise std
            std = random.uniform(0, self.std_limit)

            # Generate noise
            noise = torch.randn_like(volume) * std

            # Add noise and clip
            volume = volume + noise
            volume = torch.clamp(volume, 0.0, 1.0)

        return volume


class Compose:
    """
    Compose multiple transforms.

    Example:
        >>> transform = Compose([
        >>>     RandomHorizontalFlip(p=0.5),
        >>>     RandomRotation(degrees=10),
        >>>     RandomBrightnessContrast(p=0.5),
        >>> ])
        >>> augmented = transform(volume)
    """
    def __init__(self, transforms):
        """
        Args:
            transforms: List of transform objects
        """
        self.transforms = transforms

    def __call__(self, volume):
        """
        Apply all transforms sequentially.

        Args:
            volume: Input tensor
        Returns:
            Transformed tensor
        """
        for t in self.transforms:
            volume = t(volume)
        return volume


def get_training_augmentation(mode='medium'):
    """
    Get predefined augmentation pipeline for training.

    Args:
        mode: 'light', 'medium', or 'heavy'

    Returns:
        Compose object with augmentation transforms
    """
    if mode == 'light':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
    elif mode == 'medium':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            RandomGaussianNoise(std_limit=0.05, p=0.3),
        ])
    elif mode == 'heavy':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            RandomGaussianNoise(std_limit=0.08, p=0.5),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'light', 'medium', or 'heavy'.")


class OversamplingDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that oversamples minority classes.

    For RSNA, oversamples Moderate and Severe cases to balance classes.
    """
    def __init__(self, base_dataset, oversample_factor=5, target_classes=[1, 2]):
        """
        Args:
            base_dataset: Base dataset to wrap
            oversample_factor: How many times to repeat minority samples
            target_classes: Which classes to oversample (default: [1, 2] = Moderate, Severe)
        """
        self.base_dataset = base_dataset
        self.oversample_factor = oversample_factor
        self.target_classes = target_classes

        # Build augmented index list
        self.indices = self._build_indices()

    def _build_indices(self):
        """Build list of indices with oversampling."""
        indices = []

        # First, add all samples once
        for i in range(len(self.base_dataset)):
            indices.append(i)

        # Then, add extra copies of minority class samples
        for i in range(len(self.base_dataset)):
            _, labels = self.base_dataset[i]

            # Check if any label is in target classes
            has_minority = False
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                if labels[condition] in self.target_classes:
                    has_minority = True
                    break

            # Add extra copies
            if has_minority:
                for _ in range(self.oversample_factor - 1):
                    indices.append(i)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get item from base dataset using oversampled indices."""
        base_idx = self.indices[idx]
        return self.base_dataset[base_idx]


# Example usage
if __name__ == "__main__":
    print("Testing augmentation module...")

    # Create dummy volume
    volume = torch.rand(9, 112, 224)

    # Test individual transforms
    print("\n1. Testing RandomHorizontalFlip...")
    flip = RandomHorizontalFlip(p=1.0)
    flipped = flip(volume)
    print(f"   Original: {volume.shape}, Flipped: {flipped.shape}")

    print("\n2. Testing RandomRotation...")
    rotate = RandomRotation(degrees=10)
    rotated = rotate(volume)
    print(f"   Original: {volume.shape}, Rotated: {rotated.shape}")

    print("\n3. Testing RandomBrightnessContrast...")
    bright = RandomBrightnessContrast(p=1.0)
    adjusted = bright(volume)
    print(f"   Original range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"   Adjusted range: [{adjusted.min():.3f}, {adjusted.max():.3f}]")

    print("\n4. Testing RandomGaussianNoise...")
    noise = RandomGaussianNoise(p=1.0)
    noisy = noise(volume)
    print(f"   Original: {volume.shape}, Noisy: {noisy.shape}")

    print("\n5. Testing Compose...")
    transform = get_training_augmentation(mode='medium')
    augmented = transform(volume)
    print(f"   Original: {volume.shape}, Augmented: {augmented.shape}")

    print("\n✓ All transforms working!")
