# Reconstructor.py - Comprehensive Documentation

## Overview

This module implements differentiable CT reconstruction algorithms, specifically focusing on learnable Filtered Back Projection (FBP) methods. The implementation provides a modular, neural network-based approach to CT image reconstruction with learnable filtering components and optimized backprojection operations.

## Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Global Configuration

### Geometric Parameters
```python
n_angles = 1000        # Number of projection angles
n_detectors = 513      # Number of detector elements per projection
s_range = 0.13         # Physical detector range in meters (±0.13m)
img_size = 512         # Reconstruction grid size (512×512)
crop_size = 362        # Final output image size (362×362)
```

### Precomputed Values
```python
# Detector spacing
Delta_s = 2 * s_range / (n_detectors - 1)  # ≈ 0.000508 meters per detector

# Frequency grid for FFT filtering
freq_s = torch.fft.fftfreq(n_detectors, d=Delta_s)  # Shape: [513]
```

**Physical Interpretation**:
- **s_range**: Total detector width is 0.26m (±0.13m from center)
- **Delta_s**: Each detector element covers ~0.51mm
- **freq_s**: Frequency components for Fourier-domain filtering

---

## Core Architecture Components

### 1. LearnableFBP
**Purpose**: Main reconstruction pipeline implementing learnable Filtered Back Projection.

**Class Definition**:
```python
class LearnableFBP(nn.Module)
```

**Constructor Parameters**:
- `filtering_module` (nn.Module): Learnable frequency domain filter
- `backprojection_module` (nn.Module): Backprojection implementation
- `post_processing_module` (nn.Module): Post-reconstruction processing network

**Forward Pass Pipeline**:
```python
def forward(self, x):
    # Input shape: (B, 1, 1000, 513) - Batch of sinograms
    
    # Step 1: Transform to frequency domain
    x = torch.fft.fft(x, dim=3)  # FFT along detector dimension
    
    # Step 2: Apply learnable frequency filter
    x = self.filtering_module(x)  # Learnable filtering
    
    # Step 3: Transform back to spatial domain
    x = torch.fft.ifft(x, dim=3).real  # IFFT, keep real part
    
    # Step 4: Backproject filtered sinogram
    x = self.backprojection_module(x)  # (B, 1, 362, 362)
    
    # Step 5: Neural network post-processing
    x = self.post_processing_module(x)  # Final enhancement
    
    return x
```

**Mathematical Foundation**:
The Filtered Back Projection algorithm reconstructs images using:
```
f(x,y) = ∫₀^π ∫_{-∞}^∞ p(s,θ) * h(s-s') ds dθ
```
where:
- `p(s,θ)` is the sinogram (projection data)
- `h(s)` is the reconstruction filter
- `f(x,y)` is the reconstructed image

**Key Features**:
- **Differentiable Pipeline**: All operations support gradient computation
- **Modular Design**: Interchangeable filtering and processing components
- **Frequency Domain Processing**: Efficient convolution via FFT
- **End-to-End Training**: Learnable components integrated with traditional FBP

**Usage Example**:
```python
# Initialize components
filter_module = Filtering_Module(filter_model, window_model)
backproj_module = Vanilla_Backproj()
postproc_module = UNet(in_channels=1)  # From Modells.py

# Create reconstruction pipeline  
fbp_learnable = LearnableFBP(filter_module, backproj_module, postproc_module)

# Reconstruct CT images
sinograms = torch.randn(8, 1, 1000, 513)  # Batch of sinograms
reconstructed = fbp_learnable(sinograms)   # (8, 1, 362, 362)
```

---

### 2. Filtering_Module
**Purpose**: Learnable frequency domain filtering with separate filter and window components.

**Class Definition**:
```python
class Filtering_Module(nn.Module)
```

**Constructor Parameters**:
- `filter_model` (nn.Module): Learnable filter response (e.g., TrainableFourierSeries)
- `window_model` (nn.Module): Learnable window function (e.g., LearnableWindow)

**Forward Pass**:
```python
def forward(self, x):
    # x: Complex-valued FFT of sinogram (B, 1, 1000, 513)
    
    filtering = self.filter_model(x)    # Filter response
    window = self.window_model(x)       # Window function
    
    return window * filtering * x       # Element-wise multiplication
```

**Mathematical Interpretation**:
```
Filtered_FFT = Window(ω) × Filter(ω) × Sinogram_FFT(ω)
```

**Design Rationale**:
- **Separation of Concerns**: Filter shapes frequency response, window controls artifacts
- **Learnable Components**: Both filter and window adapt during training
- **Multiplicative Combination**: Standard practice in signal processing

**Filter Types**:
```python
# Smooth parametric filter
from Modells import TrainableFourierSeries
filter_model = TrainableFourierSeries(freq_s, init_filter, L=50)

# Direct learnable weights
from Modells import LearnableWindow
window_model = LearnableWindow(init_tensor=torch.ones(513))

# Combined filtering module
filtering = Filtering_Module(filter_model, window_model)
```

**Applications**:
- **Noise Reduction**: Low-pass characteristics suppress high-frequency noise
- **Artifact Suppression**: Window functions reduce ringing artifacts
- **Adaptive Filtering**: Learning optimal filters for specific imaging conditions

---

### 3. Vanilla_Backproj
**Purpose**: Differentiable backprojection module implementing the geometric transformation from sinogram to image space.

**Class Definition**:
```python
class Vanilla_Backproj(nn.Module)
```

**Constructor Parameters**:
- `n_angles` (int, default=1000): Number of projection angles
- `n_detectors` (int, default=513): Number of detector elements
- `s_range` (float, default=0.13): Detector physical range
- `img_size` (int, default=512): Reconstruction grid size
- `crop_size` (int, default=362): Final output size

**Geometric Precomputations**:

#### 1. Coordinate System Setup:
```python
# Image coordinate grids
x = torch.linspace(-s_range, +s_range, img_size)  # [-0.13, +0.13]
y = torch.linspace(-s_range, +s_range, img_size)  # [-0.13, +0.13]
Y, X = torch.meshgrid(x, y, indexing='ij')       # (512, 512) each

# Projection angles
theta = torch.linspace(0, torch.pi, n_angles + 1)[:-1]  # [0, π)
```

#### 2. Radon Transform Geometry:
```python
# For each angle θ and image point (x,y), compute projection coordinate s
dot_prods = X[None, :, :] * torch.cos(theta)[:, None, None] + \
            Y[None, :, :] * torch.sin(theta)[:, None, None]
# Shape: (1000, 512, 512)
```

**Mathematical Background**:
The Radon transform projects along lines defined by:
```
s = x·cos(θ) + y·sin(θ)
```
where `(s,θ)` are the sinogram coordinates.

#### 3. Detector Mapping:
```python
# Map projection coordinates to detector indices
interested_s_positions = (dot_prods + s_range) / Delta_s
# Range: [0, 512] → detector indices

# Normalize for grid_sample (requires [-1, +1] range)
grid_norm = (interested_s_positions / (n_detectors - 1)) * 2 - 1
```

**Forward Pass Implementation**:

#### Differentiable Backprojection:
```python
def forward(self, x):
    # x: Filtered sinogram (B, 1, 1000, 513)
    B = x.shape[0]
    slices = []
    
    # Process each projection angle
    for k in range(n_angles):
        # Get interpolation grid for angle k
        grid_k = self.grid_norm[k].unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Extract projection at angle k
        slice_k = x[:, :, k, :].unsqueeze(2)  # (B, 1, 1, 513)
        
        # Bilinear interpolation along projection rays
        interpolated = F.grid_sample(slice_k, grid_k, 
                                   mode='bilinear', align_corners=True)
        slices.append(interpolated)
    
    # Sum all backprojected slices
    x = torch.stack(slices, dim=1).sum(dim=1)  # (B, 1, 512, 512)
    
    # Center crop to final size
    x = self._differentiable_center_crop(x, self.crop_size)  # (B, 1, 362, 362)
    
    # Apply scaling factor
    x *= torch.pi / n_angles
    
    # Transpose for correct orientation
    return torch.transpose(x, -1, -2)
```

**Key Implementation Details**:

#### Grid Sampling:
- **F.grid_sample**: Performs bilinear interpolation using normalized coordinates
- **align_corners=True**: Ensures proper coordinate alignment
- **Mode='bilinear'**: Smooth interpolation for better gradients

#### Scaling Factor:
```python
x *= torch.pi / n_angles  # Compensates for discrete angular sampling
```

#### Cropping:
```python
@staticmethod
def _differentiable_center_crop(x, crop_size):
    """Center crop maintaining differentiability."""
    _, _, h, w = x.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return x[:, :, top:top + crop_size, left:left + crop_size]
```

**Memory and Computational Complexity**:
- **Precomputed Grids**: Stored as buffers to avoid recomputation
- **Loop Over Angles**: Memory-efficient but sequential processing
- **Grid Sample**: O(B × H × W) per angle, total O(B × H × W × n_angles)

**Usage Example**:
```python
# Initialize backprojection module
backproj = Vanilla_Backproj(n_angles=1000, n_detectors=513)

# Backproject filtered sinogram
filtered_sino = torch.randn(4, 1, 1000, 513)
reconstructed = backproj(filtered_sino)
print(f"Output shape: {reconstructed.shape}")  # (4, 1, 362, 362)
```

---

### 4. Ramp_Filter
**Purpose**: Classical ramp filter implementation for traditional FBP reconstruction.

**Class Definition**:
```python
class Ramp_Filter(nn.Module)
```

**Constructor Parameters**:
- `freqs` (torch.Tensor, default=freq_s): Frequency grid for filter definition

**Mathematical Definition**:
```python
def __init__(self, freqs=freq_s):
    super().__init__()
    self.register_buffer('ramp', torch.abs(freqs))  # |ω|
```

**Filter Response**:
The ramp filter has the frequency response:
```
H(ω) = |ω|
```

**Forward Pass**:
```python
def forward(self, x):
    # Returns filter response with proper broadcasting dimensions
    return self.ramp[None, None, None, :]  # (1, 1, 1, 513)
```

**Physical Significance**:
- **High-pass Nature**: Emphasizes high frequencies, sharpens images
- **Artifact Creation**: Can amplify noise without proper windowing
- **Classical FBP**: Standard filter in traditional reconstruction algorithms

**Frequency Response Visualization**:
```python
import matplotlib.pyplot as plt

ramp_filter = Ramp_Filter()
response = ramp_filter(None).squeeze()

plt.figure(figsize=(10, 6))
plt.plot(freq_s.numpy(), response.numpy())
plt.title('Ramp Filter Frequency Response')
plt.xlabel('Frequency (cycles/meter)')
plt.ylabel('Magnitude')
plt.grid(True)
```

**Usage in Pipeline**:
```python
# Classical FBP with ramp filter
ramp_filter = Ramp_Filter()
identity_window = LearnableWindow(init_tensor=torch.ones(513))
classical_filtering = Filtering_Module(ramp_filter, identity_window)

classical_fbp = LearnableFBP(
    filtering_module=classical_filtering,
    backprojection_module=Vanilla_Backproj(),
    post_processing_module=nn.Identity()  # No post-processing
)
```

---

### 5. CompleteReconstruct
**Purpose**: End-to-end reconstruction pipeline combining preprocessing and learnable FBP.

**Class Definition**:
```python
class CompleteReconstruct(nn.Module)
```

**Constructor Parameters**:
- `learnablefbp` (LearnableFBP): Main reconstruction pipeline
- `preprocess_net` (nn.Module): Preprocessing network for sinogram enhancement

**Forward Pass**:
```python
def forward(self, X):
    # X: Raw sinogram input (B, 1, 1000, 513)
    
    X = self.preprocess_net(X)    # Sinogram preprocessing
    X = self.learnablefbp(X)      # Learnable FBP reconstruction
    
    return X  # Final reconstructed image (B, 1, 362, 362)
```

**Pipeline Architecture**:
```
Raw Sinogram → Preprocessing Network → Learnable FBP → Reconstructed Image
    ↓                    ↓                    ↓               ↓
  Noisy Data      Enhanced Sinogram    Filtered & BP    Final Output
(B,1,1000,513)    (B,1,1000,513)     (B,1,362,362)   (B,1,362,362)
```

**Preprocessing Options**:
```python
from Modells import UNet, PseudoResnet

# Option 1: U-Net preprocessing
preprocess_unet = UNet(in_channels=1)

# Option 2: ResNet-style preprocessing  
preprocess_resnet = PseudoResnet()

# Option 3: Identity (no preprocessing)
preprocess_identity = nn.Identity()

# Create complete pipeline
complete_recon = CompleteReconstruct(
    learnablefbp=fbp_pipeline,
    preprocess_net=preprocess_unet
)
```

**Training Strategy**:
```python
def train_complete_reconstruction():
    # Joint optimization of preprocessing and reconstruction
    optimizer = torch.optim.Adam(complete_recon.parameters(), lr=1e-4)
    
    for batch_idx, (raw_sinograms, ground_truth) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # End-to-end forward pass
        reconstructed = complete_recon(raw_sinograms)
        
        # Combined loss
        recon_loss = F.mse_loss(reconstructed, ground_truth)
        loss = recon_loss
        
        loss.backward()
        optimizer.step()
```

---

## Mathematical Foundations

### 1. Filtered Back Projection Theory

#### Forward Radon Transform:
```
R_θ[f](s) = ∫∫ f(x,y) δ(x·cos(θ) + y·sin(θ) - s) dx dy
```

#### Inverse Radon Transform (FBP):
```
f(x,y) = 1/(2π) ∫₀^π ∫_{-∞}^∞ R_θ(s) |ω| e^{i2πωs} e^{-i2πω(x·cos(θ)+y·sin(θ))} dω ds dθ
```

#### Discrete Implementation:
```
f[i,j] ≈ (π/N_θ) Σ_{k=0}^{N_θ-1} Σ_{n=0}^{N_s-1} p[k,n] h[n] sinc(s_{i,j,k} - n)
```

where:
- `p[k,n]` is the discrete sinogram
- `h[n]` is the discrete filter
- `s_{i,j,k}` is the projection coordinate for pixel (i,j) at angle k

### 2. Frequency Domain Filtering

#### Convolution Theorem:
```
IFFT(FFT(sinogram) × FFT(filter)) = sinogram ⊛ filter
```

#### Filter Design Criteria:
- **Ramp Filter**: `H(ω) = |ω|` for exact reconstruction
- **Windowed Filters**: `H(ω) = |ω| × W(ω)` for noise control
- **Learnable Filters**: `H(ω) = F_θ(ω)` optimized via gradient descent

### 3. Geometric Transformations

#### Coordinate Mapping:
```python
# From image coordinates (x,y) to sinogram coordinates (s,θ)
s = x·cos(θ) + y·sin(θ)

# From sinogram coordinates to detector index
detector_idx = (s + s_range) / Delta_s
```

#### Grid Sample Normalization:
```python
# PyTorch grid_sample requires coordinates in [-1, +1]
normalized_coord = (detector_idx / (n_detectors - 1)) * 2 - 1
```

---

## Performance Optimization

### 1. Memory Management
```python
class MemoryEfficientBackproj(Vanilla_Backproj):
    def forward(self, x):
        # Process angles in chunks to reduce memory
        chunk_size = 100
        B = x.shape[0]
        accumulated = torch.zeros(B, 1, self.crop_size, self.crop_size, 
                                 device=x.device, dtype=x.dtype)
        
        for start_idx in range(0, n_angles, chunk_size):
            end_idx = min(start_idx + chunk_size, n_angles)
            chunk_result = self._process_angle_chunk(x, start_idx, end_idx)
            accumulated += chunk_result
        
        return accumulated * torch.pi / n_angles
```

### 2. GPU Optimization
```python
def optimize_for_gpu():
    # Use half precision for memory efficiency
    model = LearnableFBP(...).half()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing = True
    
    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model
```

### 3. Compile Optimization (PyTorch 2.0+)
```python
# Compile for faster execution
@torch.compile
class CompiledBackproj(Vanilla_Backproj):
    pass

compiled_model = CompiledBackproj()
```

---

## Advanced Usage Patterns

### 1. Multi-Resolution Reconstruction
```python
class MultiResolutionFBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_res_fbp = LearnableFBP(...)  # 128x128 output
        self.high_res_fbp = LearnableFBP(...) # 362x362 output
        
    def forward(self, x):
        # Coarse reconstruction
        low_res = self.low_res_fbp(x)
        
        # Refine with high resolution
        high_res = self.high_res_fbp(x)
        
        # Combine using learned weights
        return self.combine_resolutions(low_res, high_res)
```

### 2. Iterative Reconstruction
```python
class IterativeFBP(nn.Module):
    def __init__(self, n_iterations=5):
        super().__init__()
        self.n_iterations = n_iterations
        self.fbp = LearnableFBP(...)
        self.update_net = UNet(in_channels=2)  # Current + residual
        
    def forward(self, sinogram):
        reconstruction = self.fbp(sinogram)
        
        for i in range(self.n_iterations):
            # Forward project current reconstruction
            forward_proj = self.radon_transform(reconstruction)
            
            # Compute residual
            residual = sinogram - forward_proj
            
            # Update reconstruction
            input_concat = torch.cat([reconstruction, residual], dim=1)
            update = self.update_net(input_concat)
            reconstruction = reconstruction + update
            
        return reconstruction
```

### 3. Uncertainty Quantification
```python
class BayesianFBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = LearnableFBP(...)
        self.variance_net = LearnableFBP(...)
        
    def forward(self, x):
        mean = self.mean_net(x)
        log_variance = self.variance_net(x)
        
        if self.training:
            # Sample from posterior during training
            std = torch.exp(0.5 * log_variance)
            epsilon = torch.randn_like(std)
            return mean + epsilon * std, log_variance
        else:
            return mean, log_variance
```

---

## Integration Examples

### 1. Training Pipeline
```python
def train_learnable_fbp():
    # Initialize components
    filter_model = TrainableFourierSeries(freq_s, init_filter, L=50)
    window_model = LearnableWindow()
    filtering = Filtering_Module(filter_model, window_model)
    backproj = Vanilla_Backproj()
    postproc = UNet(in_channels=1)
    
    model = LearnableFBP(filtering, backproj, postproc)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch_idx, (sinograms, ground_truth) in enumerate(train_loader):
            optimizer.zero_grad()
            
            reconstructed = model(sinograms)
            loss = F.mse_loss(reconstructed, ground_truth)
            
            loss.backward()
            optimizer.step()
```

### 2. Evaluation and Visualization
```python
def evaluate_reconstruction(model, test_loader):
    model.eval()
    with torch.no_grad():
        for sinograms, ground_truth in test_loader:
            reconstructed = model(sinograms)
            
            # Compute metrics
            psnr_val = psnr(reconstructed, ground_truth)
            ssim_val = ssim_metric(reconstructed, ground_truth)
            
            # Visualize results
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(sinograms[0, 0].cpu(), aspect='auto')
            plt.title('Input Sinogram')
            
            plt.subplot(132)  
            plt.imshow(reconstructed[0, 0].cpu(), cmap='gray')
            plt.title(f'Reconstruction (PSNR: {psnr_val:.2f})')
            
            plt.subplot(133)
            plt.imshow(ground_truth[0, 0].cpu(), cmap='gray')  
            plt.title('Ground Truth')
            
            plt.tight_layout()
            plt.show()
```

### 3. Model Comparison
```python
def compare_reconstruction_methods():
    # Classical FBP
    classical = LearnableFBP(
        filtering_module=Filtering_Module(Ramp_Filter(), LearnableWindow()),
        backprojection_module=Vanilla_Backproj(),
        post_processing_module=nn.Identity()
    )
    
    # Learnable FBP
    learnable = LearnableFBP(
        filtering_module=Filtering_Module(TrainableFourierSeries(...), LearnableWindow()),
        backprojection_module=Vanilla_Backproj(),
        post_processing_module=UNet(in_channels=1)
    )
    
    # End-to-end learning
    end_to_end = UNet(in_channels=1)  # Direct sinogram to image
    
    # Compare on test set
    methods = {'Classical FBP': classical, 'Learnable FBP': learnable, 'End-to-End': end_to_end}
    results = {}
    
    for name, model in methods.items():
        psnr_scores = []
        ssim_scores = []
        
        for sinograms, ground_truth in test_loader:
            if name == 'End-to-End':
                output = model(sinograms)
            else:
                output = model(sinograms)
                
            psnr_scores.append(psnr(output, ground_truth).item())
            ssim_scores.append(ssim_metric(output, ground_truth).item())
        
        results[name] = {
            'PSNR': np.mean(psnr_scores),
            'SSIM': np.mean(ssim_scores)
        }
    
    return results
```

This comprehensive documentation covers all aspects of the Reconstructor.py module, providing both theoretical understanding and practical implementation guidance for differentiable CT reconstruction systems.
