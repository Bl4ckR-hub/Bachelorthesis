# CT_library.py - Comprehensive Documentation

## Overview

This module provides essential utilities for Computed Tomography (CT) image processing and reconstruction, specifically designed for the Low Dose Parallel Beam (LoDoPaB) dataset. It includes data loading, image preprocessing, normalization functions, and Radon transform operations for CT reconstruction tasks.

## Dependencies

```python
import os
from torch.utils.data import Dataset
import torch
import h5py
import odl  # Operator Discretization Library for tomographic operations
import torch.nn.functional as F
import numpy as np
```

## Core Components

### 1. Dataset Management

#### LoDoPaB_Dataset
**Purpose**: PyTorch Dataset class for loading and managing the Low Dose Parallel Beam (LoDoPaB) CT dataset.

**Class Definition**:
```python
class LoDoPaB_Dataset(Dataset)
```

**Constructor Parameters**:
- `sino_dir` (str): Directory path containing sinogram observation files
- `gt_images_dir` (str): Directory path containing ground truth image files
- `transform` (callable, optional): Transform to apply to sinogram data
- `target_transform` (callable, optional): Transform to apply to ground truth images
- `suffix` (str, optional): File suffix filter for specific dataset subsets
- `amount_images` (int, optional): Maximum number of images to load (for dataset size limitation)

**Key Features**:
- **Automatic File Discovery**: Scans directories for files containing 'ground_truth' and 'observation' keywords
- **HDF5 Support**: Efficiently loads data from HDF5 files using h5py
- **Index Mapping**: Creates internal mapping for efficient data access across multiple files
- **Flexible Filtering**: Supports suffix-based filtering and dataset size limitation
- **Memory Efficient**: Loads data on-demand rather than storing everything in memory

**Data Structure Assumptions**:
- Each HDF5 file contains 128 slices (except possibly the last file)
- Data is stored under the 'data' key in HDF5 files
- Sinogram and ground truth files are paired and sorted consistently

**Methods**:

##### `__len__(self)`
**Returns**: Total number of available image slices across all files

##### `__getitem__(self, idx)`
**Parameters**:
- `idx` (int): Index of the sample to retrieve

**Returns**: 
- `tuple`: (sinogram, ground_truth_image)
  - sinogram: `torch.Tensor` of shape (1, H, W)
  - ground_truth_image: `torch.Tensor` of shape (1, H, W)

**Process**:
1. Maps linear index to (file_index, data_index) pair
2. Opens corresponding HDF5 files
3. Loads specific slice from each file
4. Converts numpy arrays to PyTorch tensors
5. Adds channel dimension for compatibility
6. Applies transforms if specified

**Usage Example**:
```python
dataset = LoDoPaB_Dataset(
    sino_dir="/path/to/sinograms",
    gt_images_dir="/path/to/ground_truth",
    suffix="train",
    amount_images=1000
)

# Access data
sinogram, gt_image = dataset[0]
print(f"Sinogram shape: {sinogram.shape}")  # (1, H, W)
print(f"GT image shape: {gt_image.shape}")  # (1, H, W)
```

---

### 2. Image Processing Functions

#### crop_zoom_top_left
**Purpose**: Extracts a rectangular region from an image tensor using top-left corner coordinates.

**Function Signature**:
```python
def crop_zoom_top_left(image: torch.Tensor, x: int, y: int, width: int, height: int) -> torch.Tensor
```

**Parameters**:
- `image` (torch.Tensor): Input image tensor
  - 2D: Shape (H, W) for grayscale images
  - 3D: Shape (C, H, W) for multi-channel images
- `x` (int): x-coordinate (column) of the top-left corner
- `y` (int): y-coordinate (row) of the top-left corner  
- `width` (int): Width of the cropped region
- `height` (int): Height of the cropped region

**Returns**: 
- `torch.Tensor`: Cropped image tensor maintaining original dimensionality

**Coordinate System**:
```
(0,0) ────────── x →
│
│     (x,y) ┌─────────┐
│           │  crop   │
│           │ region  │
y           └─────────┘
↓
```

**Usage Examples**:
```python
# 2D grayscale image
img_2d = torch.randn(512, 512)
cropped_2d = crop_zoom_top_left(img_2d, x=100, y=100, width=200, height=200)
# Result shape: (200, 200)

# 3D multi-channel image
img_3d = torch.randn(3, 512, 512)
cropped_3d = crop_zoom_top_left(img_3d, x=50, y=50, width=100, height=100)
# Result shape: (3, 100, 100)
```

---

### 3. Normalization Functions

#### min_max_normalize
**Purpose**: Performs min-max normalization on batch tensors, scaling values to [0, 1] range per sample.

**Function Signature**:
```python
def min_max_normalize(x, eps=1e-8) -> torch.Tensor
```

**Parameters**:
- `x` (torch.Tensor): Input tensor of shape (B, 1, H, W) or (B, H, W)
- `eps` (float, default=1e-8): Small epsilon value for numerical stability

**Mathematical Formula**:
```
x_normalized = (x - x_min) / (x_max - x_min + ε)
```

**Key Features**:
- **Per-Sample Normalization**: Min/max calculated independently for each sample in batch
- **Preserves Spatial Structure**: Maintains relative intensity relationships within each image
- **Numerical Stability**: Epsilon prevents division by zero

**Usage Example**:
```python
# Batch of CT images
batch = torch.randn(8, 1, 362, 362)  # 8 images, 1 channel, 362x362
normalized = min_max_normalize(batch)
print(f"Original range: [{batch.min():.3f}, {batch.max():.3f}]")
print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
# Output: Normalized range: [0.000, 1.000]
```

#### min_max_norm
**Purpose**: Alternative min-max normalization with different tensor dimension handling.

**Function Signature**:
```python
def min_max_norm(X, eps=1e-9) -> torch.Tensor
```

**Parameters**:
- `X` (torch.Tensor): Input tensor
- `eps` (float, default=1e-9): Numerical stability constant

**Implementation Details**:
- Computes min/max values across spatial dimensions (dim=2)
- Uses unsqueeze operations for broadcasting compatibility
- Suitable for 3D tensors with shape (B, C, H, W)

#### norm_to_classic
**Purpose**: Inverse normalization function to restore original value ranges.

**Function Signature**:
```python
def norm_to_classic(X, v_min, v_max, eps=1e-9) -> torch.Tensor
```

**Parameters**:
- `X` (torch.Tensor): Normalized tensor (typically in [0,1] range)
- `v_min` (torch.Tensor): Original minimum values
- `v_max` (torch.Tensor): Original maximum values
- `eps` (float, default=1e-9): Numerical stability constant

**Mathematical Formula**:
```
X_restored = X * (v_max - v_min + ε) + v_min
```

**Usage Pattern**:
```python
# Forward normalization
original = torch.randn(4, 1, 256, 256)
v_min = original.min()
v_max = original.max()
normalized = min_max_norm(original)

# Inverse normalization
restored = norm_to_classic(normalized, v_min, v_max)
assert torch.allclose(original, restored, atol=1e-6)
```

---

### 4. CT-Specific Transformation Functions

#### gt_to_coeffs
**Purpose**: Converts ground truth images to attenuation coefficients by scaling.

**Function Signature**:
```python
def gt_to_coeffs(Y, max=81.35858) -> torch.Tensor
```

**Parameters**:
- `Y` (torch.Tensor): Normalized ground truth image
- `max` (float, default=81.35858): Maximum attenuation coefficient value

**Physical Background**:
- CT images represent linear attenuation coefficients
- The scaling factor (81.35858) represents the maximum expected attenuation coefficient
- This conversion is essential for physics-based CT reconstruction

#### X_to_minuspostlog
**Purpose**: Converts normalized images to minus-post-log domain for CT processing.

**Function Signature**:
```python
def X_to_minuspostlog(X, max=81.35858) -> torch.Tensor
```

**Parameters**:
- `X` (torch.Tensor): Input image tensor
- `max` (float, default=81.35858): Maximum value for scaling

**Mathematical Background**:
In CT reconstruction, the post-log domain represents:
```
post_log = log(I₀/I)
```
where I₀ is incident intensity and I is transmitted intensity.

#### minuspostlog_to_proj
**Purpose**: Converts minus-post-log values to projection measurements.

**Function Signature**:
```python
def minuspostlog_to_proj(X, N_0=4096) -> torch.Tensor
```

**Parameters**:
- `X` (torch.Tensor): Minus-post-log values
- `N_0` (float, default=4096): Incident photon count

**Mathematical Formula**:
```
projection = N₀ * exp(-X)
```

**Physical Interpretation**:
- N₀ represents the number of incident photons
- exp(-X) gives the transmission fraction
- Result represents detected photon counts

**CT Processing Pipeline**:
```python
# Typical CT processing workflow
gt_image = torch.randn(1, 1, 362, 362)  # Normalized GT
coeffs = gt_to_coeffs(gt_image)         # Scale to physical units
minus_postlog = X_to_minuspostlog(coeffs)  # Convert to log domain
projections = minuspostlog_to_proj(minus_postlog)  # Get measurements
```

---

### 5. Radon Transform Class

#### RadonTransform
**Purpose**: Implements the Radon transform for CT image reconstruction using the ODL library.

**Class Definition**:
```python
class RadonTransform()
```

**Constructor Parameters**:
- `device` (torch.device, default=torch.device('cpu')): Computation device

**Geometric Configuration**:

##### Image Space
- **Domain**: [-0.13, +0.13] × [-0.13, +0.13] meters
- **Resolution**: 1000 × 1000 pixels
- **Pixel Size**: 0.26mm × 0.26mm

##### Projection Geometry
- **Type**: Parallel beam geometry
- **Angular Range**: 0 to π radians (180°)
- **Number of Angles**: 1000 projections
- **Detector Length**: √(0.26² + 0.26²) ≈ 0.368 meters
- **Detector Bins**: 513 detector elements

**Mathematical Background**:
The Radon transform R maps a function f(x,y) to its line integrals:
```
(Rf)(s,θ) = ∫∫ f(x,y) δ(x·cos(θ) + y·sin(θ) - s) dx dy
```

**Implementation Details**:
- Uses ASTRA toolbox backend for efficient computation
- Automatically selects CPU or CUDA implementation based on device
- Configured for medical CT imaging parameters

#### Methods

##### `radon(self, Y_coeffs)`
**Purpose**: Computes forward Radon transform of attenuation coefficient images.

**Parameters**:
- `Y_coeffs` (torch.Tensor): Batch of attenuation coefficient images

**Process**:
1. **Interpolation**: Upsamples input to 1000×1000 resolution
2. **CPU Transfer**: Moves data to CPU for ODL processing
3. **NumPy Conversion**: Converts PyTorch tensors to NumPy arrays
4. **Radon Transform**: Applies forward projection for each image in batch
5. **Tensor Conversion**: Returns results as PyTorch tensors

**Input/Output Shapes**:
- Input: (B, 1, H, W) - Batch of images
- Output: (B, 1, 1000, 513) - Batch of sinograms

**Usage Example**:
```python
# Initialize Radon transform
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
radon_op = RadonTransform(device=device)

# Generate synthetic phantom
phantom = torch.zeros(1, 1, 362, 362)
phantom[0, 0, 150:250, 150:250] = 1.0  # Square phantom

# Convert to attenuation coefficients
coeffs = gt_to_coeffs(phantom)

# Compute sinogram
sinogram = radon_op.radon(coeffs)
print(f"Sinogram shape: {sinogram.shape}")  # (1, 1, 1000, 513)
```

**Performance Considerations**:
- **Memory Usage**: Large intermediate arrays (1000×1000) require significant memory
- **Computation Time**: Forward projection is computationally intensive
- **Device Management**: Automatic handling of CPU/GPU transfers
- **Batch Processing**: Processes each image individually in a loop

**Geometric Accuracy**:
The transform is configured to match standard CT imaging setups:
- **Fan-to-parallel conversion**: Uses parallel beam approximation
- **Calibrated geometry**: Detector size matches image field of view
- **Angular sampling**: Dense angular sampling (1000 angles) for high quality

---

## Physical and Mathematical Context

### CT Imaging Physics
1. **X-ray Attenuation**: X-rays are attenuated according to Beer's law
2. **Line Integrals**: Each detector measurement represents a line integral
3. **Radon Transform**: Mathematical foundation for CT reconstruction
4. **Inverse Problem**: Reconstruction involves inverting the Radon transform

### Data Domains
1. **Image Domain**: Attenuation coefficients μ(x,y)
2. **Projection Domain**: Line integral measurements
3. **Log Domain**: Logarithmic transformation of measurements
4. **Normalized Domain**: Scaled values for neural network processing

### Mathematical Relationships
```
I = I₀ * exp(-∫ μ(x,y) dl)        # Beer's law
p = log(I₀/I) = ∫ μ(x,y) dl      # Log transformation (post-log)
sinogram = Radon(μ)               # Forward projection
μ_reconstructed = Radon⁻¹(sinogram) # Reconstruction
```

---

## Integration with Neural Networks

### Dataset Integration
```python
from torch.utils.data import DataLoader

# Create dataset
dataset = LoDoPaB_Dataset(
    sino_dir="data/sinograms",
    gt_images_dir="data/ground_truth",
    transform=min_max_normalize,
    target_transform=min_max_normalize
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for sinograms, gt_images in dataloader:
    # sinograms: (32, 1, H, W)
    # gt_images: (32, 1, H, W)
    predictions = model(sinograms)
    loss = criterion(predictions, gt_images)
```

### Physics-Informed Training
```python
# Example: Consistency loss using Radon transform
radon_op = RadonTransform(device='cuda')

def physics_consistency_loss(prediction, target):
    # Convert to physical domain
    pred_coeffs = gt_to_coeffs(prediction)
    target_coeffs = gt_to_coeffs(target)
    
    # Compute forward projections
    pred_sino = radon_op.radon(pred_coeffs)
    target_sino = radon_op.radon(target_coeffs)
    
    # Consistency loss in sinogram domain
    return F.mse_loss(pred_sino, target_sino)
```

---

## Error Handling and Edge Cases

### Dataset Loading
- **Missing Files**: Graceful handling of missing sinogram or ground truth files
- **Inconsistent Sizes**: Validation of matching file counts
- **HDF5 Errors**: Proper exception handling for corrupted files

### Image Processing
- **Dimension Validation**: Checks for 2D/3D tensor compatibility
- **Boundary Conditions**: Crop operations respect image boundaries
- **Numerical Stability**: Epsilon values prevent division by zero

### Radon Transform
- **Device Compatibility**: Automatic fallback from CUDA to CPU
- **Memory Management**: Efficient handling of large intermediate arrays
- **Interpolation Artifacts**: High-quality bilinear interpolation

---

## Performance Optimization

### Memory Efficiency
- **Lazy Loading**: Dataset loads images on-demand
- **Batch Processing**: Efficient batch operations for normalization
- **Device Management**: Automatic CPU/GPU memory handling

### Computational Efficiency
- **Vectorized Operations**: All functions use PyTorch's vectorized operations
- **ASTRA Integration**: Leverages optimized tomographic reconstruction library
- **NumPy Integration**: Efficient array operations where needed

### Best Practices
```python
# Recommended usage patterns
def efficient_preprocessing(batch):
    # Normalize in-place when possible
    normalized = min_max_normalize(batch)
    
    # Use appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalized = normalized.to(device)
    
    return normalized

# Memory-conscious dataset loading
dataset = LoDoPaB_Dataset(
    sino_dir="data/sinograms",
    gt_images_dir="data/ground_truth",
    amount_images=10000  # Limit dataset size for development
)
```

---

## Dependencies and Installation

### Required Packages
```bash
pip install torch torchvision
pip install h5py
pip install numpy
pip install odl[astra]  # For tomographic operations
```

### ODL Installation Notes
- ODL (Operator Discretization Library) requires ASTRA toolbox
- ASTRA provides efficient CT reconstruction algorithms
- GPU support requires CUDA-compatible installation

### Compatibility
- **PyTorch**: Compatible with PyTorch 1.0+
- **Python**: Requires Python 3.6+
- **CUDA**: Optional for GPU acceleration
- **Operating Systems**: Linux, Windows, macOS (with appropriate ASTRA installation)

This comprehensive documentation covers all aspects of the CT_library.py module, providing both theoretical background and practical usage examples for CT image processing and reconstruction tasks.
