# Criterion.py - Comprehensive Documentation

## Overview

This module provides a comprehensive collection of advanced loss functions specifically designed for medical image processing, particularly CT (Computed Tomography) image reconstruction and enhancement. The loss functions are implemented in PyTorch and focus on preserving various image characteristics such as frequency information, structural details, edges, and smoothness.

## Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import CT_library  # Custom CT processing library
```

## Loss Function Categories

### 1. Frequency Domain Loss Functions

#### FFTBandLoss
**Purpose**: Focuses on high-frequency components by masking out low-frequency information in the FFT domain.

**Class Definition**:
```python
class FFTBandLoss(nn.Module)
```

**Parameters**:
- `low_freq_ratio` (float, default=0.2): Ratio of low-frequency band to mask out (0.2 means central 20% of frequencies are ignored)
- `loss_fn` (nn.Module, default=nn.L1Loss()): Base loss function to apply on frequency components

**Key Methods**:
- `create_high_freq_mask(shape, device)`: Creates circular mask to isolate high frequencies
- `forward(pred, target)`: Computes FFT-based loss on high-frequency components

**Usage Example**:
```python
loss_fft = FFTBandLoss(low_freq_ratio=0.2)
pred_image = torch.randn(4, 3, 256, 256)
target_image = torch.randn(4, 3, 256, 256)
loss = loss_fft(pred_image, target_image)
```

**Mathematical Background**:
- Applies 2D FFT to input images
- Shifts zero-frequency component to center using `fftshift`
- Creates circular mask to filter low frequencies
- Computes loss on magnitude of high-frequency components

---

#### LaplacianPyramidLoss
**Purpose**: Encourages similarity across multiple frequency bands using Laplacian pyramid decomposition.

**Class Definition**:
```python
class LaplacianPyramidLoss(nn.Module)
```

**Parameters**:
- `max_levels` (int, default=3): Number of pyramid levels
- `kernel_size` (int, default=5): Size of Gaussian kernel
- `sigma` (float, default=1.0): Standard deviation for Gaussian kernel
- `loss_fn` (nn.Module, default=nn.L1Loss()): Base loss function

**Key Methods**:
- `create_gaussian_kernel(kernel_size, sigma)`: Creates 2D Gaussian kernel for blurring
- `laplacian_pyramid(img, levels)`: Builds Laplacian pyramid decomposition
- `forward(pred, target)`: Computes weighted loss across all pyramid levels

**Mathematical Background**:
- Constructs Laplacian pyramid by iterative blur, downsample, upsample operations
- Each level captures different frequency bands
- Uses exponential weighting (2^level) to emphasize finer details

---

#### GaussianEdgeEnhancedLoss
**Purpose**: Emphasizes edge preservation using Gaussian high-pass filtering in frequency domain.

**Parameters**:
- `cutoff_freq` (float, default=0.5): Cutoff frequency for high-pass filter
- `sigma` (int, default=1): Standard deviation for Gaussian filter
- `img_size` (int, default=362): Image size for pre-computed filter

**Mathematical Formula**:
```
H(u,v) = 1 - exp(-(D(u,v) - cutoff_freq)² / (2σ²))
```
where D(u,v) is the distance from DC component.

---

### 2. Gradient and Edge Preservation Loss Functions

#### GradientEdgeLoss
**Purpose**: Preserves edge information by comparing gradient maps using Sobel operators.

**Features**:
- Uses Sobel kernels for horizontal and vertical gradient computation
- Automatic grayscale conversion for multi-channel images
- L1 loss comparison between gradient maps

**Sobel Kernels**:
- Horizontal: `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`
- Vertical: `[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]`

---

#### GradientVarianceLoss
**Purpose**: Compares local gradient variance patterns between images.

**Parameters**:
- `n` (int, default=4): Patch size for variance calculation

**Process**:
1. Compute gradient maps using Sobel operators
2. Divide gradient maps into n×n non-overlapping patches
3. Calculate variance within each patch
4. Compare variance maps using MSE loss

---

#### MultiScaleGradientVarianceLoss
**Purpose**: Multi-scale version of gradient variance loss for robust edge preservation.

**Parameters**:
- `scales` (list, default=[1, 0.5, 0.25]): Different scales for analysis
- `weights` (list, optional): Weights for each scale
- `patch_size` (int, default=4): Patch size for variance calculation

**Process**:
- Applies gradient variance loss at multiple image scales
- Combines losses with weighted average
- Provides scale-invariant edge preservation

---

### 3. Structural Preservation Loss Functions

#### TotalVariationLoss
**Purpose**: Encourages spatial smoothness by penalizing large gradients.

**Mathematical Formula**:
```
TV(I) = Σ|I(x+1,y) - I(x,y)| + Σ|I(x,y+1) - I(x,y)|
```

**Use Cases**:
- Noise reduction
- Smoothness regularization
- Artifact suppression

---

#### SinoLocalStrucLoss
**Purpose**: Preserves local structural information using second-order derivatives.

**Mathematical Background**:
Uses Hessian matrix components:
- `dxx`: Second derivative w.r.t. x
- `dyy`: Second derivative w.r.t. y  
- `dxy`, `dyx`: Mixed derivatives

**Loss Formula**:
```
Loss = √(Dxx² + Dyy² + Dxy² + Dyx²)
```
where D represents differences between target and output derivatives.

---

### 4. Perceptual and Feature-Based Loss Functions

#### PerceptualLoss
**Purpose**: Uses pre-trained VGG16 features for perceptual similarity.

**Parameters**:
- `layers` (list, default=[3, 8, 15]): VGG16 layer indices for feature extraction
- `weight` (float, default=1.0): Loss weight multiplier

**Features**:
- Converts single-channel CT images to 3-channel RGB
- Extracts features from multiple VGG16 layers
- Computes L1 loss between feature maps

---

### 5. Sinogram-Specific Loss Functions

#### SinoMAP
**Purpose**: Maximum A Posteriori (MAP) estimation loss for sinogram reconstruction.

**Parameters**:
- `sigma` (float, default=10): Noise standard deviation

**Mathematical Background**:
Implements statistical reconstruction model:
```
Loss = (X_gt_proj - X_proj)²/(2σ²) + X_proj*(-X_gt_minuspostlog - log) + log(X_proj)
```

---

#### WeightedL1L2SinogramLoss
**Purpose**: Weighted combination of L1 and L2 losses for sinogram reconstruction.

**Parameters**:
- `N0` (float, default=4096): Incident photon count
- `sigma_e` (float, default=0.0): Electronic noise standard deviation
- `lambda_wls` (float, default=1.0): Weight for WLS term
- `lambda_l1` (float, default=0.1): Weight for L1 term
- `eps` (float, default=1e-6): Numerical stability constant

**Components**:
1. **Weighted Least Squares (WLS)**: Accounts for Poisson noise statistics
2. **L1 Regularization**: Promotes sparsity and edge preservation

---

### 6. Utility Functions and Derivative Operators

#### Second-Order Derivative Functions

**dxx(img)**: Second derivative w.r.t. x-axis
```python
kernel = [[0, 0, 0],
          [1, -2, 1], 
          [0, 0, 0]]
```

**dyy(img)**: Second derivative w.r.t. y-axis
```python
kernel = [[0, 1, 0],
          [0, -2, 0],
          [0, 1, 0]]
```

**dxy(img)**: Mixed derivative ∂²f/∂x∂y
- Sequential application of y-derivative then x-derivative

**dyx(img)**: Mixed derivative ∂²f/∂y∂x
- Sequential application of x-derivative then y-derivative
- Alternative implementation using single convolution kernel

---

### 7. Composite Loss Functions

#### MultipleLoss
**Purpose**: Combines multiple loss functions with specified weights.

**Parameters**:
- `losses` (list): List of loss function instances
- `weights` (list): Corresponding weights for each loss

**Usage**:
```python
losses = [nn.L1Loss(), nn.MSELoss(), FFTBandLoss()]
weights = [1.0, 0.5, 0.2]
combined_loss = MultipleLoss(losses, weights)
```

---

#### fft_loss (Standalone Function)
**Purpose**: Simple FFT-based loss function for magnitude spectrum comparison.

**Parameters**:
- `pred` (torch.Tensor): Predicted image (B, 1, H, W)
- `target` (torch.Tensor): Target image (B, 1, H, W)
- `weight` (float, default=1.0): Loss weight

**Formula**:
```
Loss = weight * mean(|abs(FFT(pred)) - abs(FFT(target))|)
```

---

## Usage Recommendations

### For CT Image Reconstruction:
1. **Primary Loss**: L1 or L2 loss for basic reconstruction
2. **Edge Preservation**: GradientEdgeLoss or GradientVarianceLoss
3. **Frequency Content**: FFTBandLoss for high-frequency details
4. **Smoothness**: TotalVariationLoss for noise reduction
5. **Perceptual Quality**: PerceptualLoss for visual similarity

### For Sinogram Processing:
1. **Statistical Modeling**: SinoMAP for MAP estimation
2. **Noise Handling**: WeightedL1L2SinogramLoss
3. **Structure Preservation**: SinoLocalStrucLoss

### Typical Combination:
```python
# Example multi-objective loss
losses = [
    nn.L1Loss(),                    # Basic reconstruction
    GradientEdgeLoss(),            # Edge preservation
    FFTBandLoss(low_freq_ratio=0.3), # High-freq details
    TotalVariationLoss()           # Smoothness
]
weights = [1.0, 0.5, 0.3, 0.1]
final_loss = MultipleLoss(losses, weights)
```

## Performance Considerations

1. **Memory Usage**: Frequency domain operations require additional memory
2. **Computational Cost**: Multi-scale and pyramid methods are computationally intensive
3. **GPU Compatibility**: All functions support CUDA tensors
4. **Gradient Flow**: All loss functions maintain proper gradient computation

## Mathematical Foundations

The loss functions are based on several mathematical principles:

1. **Fourier Analysis**: FFT-based losses leverage frequency domain representations
2. **Differential Geometry**: Gradient and Hessian-based losses use local geometric properties
3. **Scale-Space Theory**: Multi-scale approaches provide scale-invariant analysis
4. **Statistical Modeling**: Sinogram losses incorporate noise statistics
5. **Perceptual Models**: VGG-based losses utilize learned feature representations

## Integration with CT_library

Several functions depend on the custom `CT_library` module for:
- `X_to_minuspostlog()`: Conversion from image to minus-post-log domain
- `minuspostlog_to_proj()`: Conversion to projection domain

Ensure proper CT_library implementation for sinogram-specific losses.
