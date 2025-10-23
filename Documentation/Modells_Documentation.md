# Modells.py - Comprehensive Documentation

## Overview

This module provides a comprehensive collection of neural network architectures specifically designed for medical image processing, particularly CT reconstruction and enhancement tasks. The module includes various U-Net implementations, specialized filtering components, and custom architectural blocks optimized for different medical imaging scenarios.

## Dependencies

```python
import torch.nn as nn
import torch
```

## Architecture Categories

### 1. Utility Components

#### Clamper
**Purpose**: A simple neural network module that clamps input values to a specified range.

**Class Definition**:
```python
class Clamper(nn.Module)
```

**Constructor Parameters**:
- `min_val` (float): Minimum value for clamping
- `max_val` (float): Maximum value for clamping

**Mathematical Operation**:
```
output = clamp(input, min_val, max_val)
```

**Use Cases**:
- **Output Range Control**: Ensuring model outputs stay within valid ranges
- **Gradient Clipping**: Preventing exploding gradients
- **Physical Constraints**: Enforcing realistic value ranges for medical images
- **Normalization**: Maintaining consistent data ranges

**Usage Example**:
```python
# Clamp CT values to valid range
clamper = Clamper(min_val=0.0, max_val=1.0)
output = model(input)
clamped_output = clamper(output)  # Values now in [0, 1]

# For 8-bit image output
clamper_8bit = Clamper(min_val=0, max_val=255)
```

---

### 2. Basic Building Blocks

#### CNNBlock
**Purpose**: Fundamental convolutional building block with double convolution and ReLU activation.

**Class Definition**:
```python
class CNNBlock(nn.Module)
```

**Constructor Parameters**:
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels

**Architecture**:
```
Input → Conv2d(3×3, padding=1) → ReLU → Conv2d(3×3, padding=1) → ReLU → Output
```

**Key Characteristics**:
- **Kernel Size**: 3×3 convolutions for local feature extraction
- **Padding**: Same padding (padding=1) preserves spatial dimensions
- **Stride**: 1 for no downsampling within the block
- **Activation**: ReLU for non-linearity and computational efficiency
- **Double Convolution**: Standard pattern for increased receptive field

**Receptive Field**: Each CNNBlock increases receptive field by 4 pixels (2 per convolution)

**Usage Example**:
```python
# Basic feature extraction
cnn_block = CNNBlock(in_channels=64, out_channels=128)
features = cnn_block(input_tensor)  # Shape: (B, 128, H, W)

# Calculate output dimensions (same as input due to padding=1)
# If input is (B, 64, 256, 256), output is (B, 128, 256, 256)
```

---

#### EncoderBlock
**Purpose**: Encoder component for U-Net architecture with convolution followed by downsampling.

**Class Definition**:
```python
class EncoderBlock(nn.Module)
```

**Constructor Parameters**:
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels

**Architecture**:
```
Input → CNNBlock → [Skip Connection] → MaxPool2d(2×2) → Output
      ↓
   X_old (for skip connections)
```

**Output**:
- `X_old` (torch.Tensor): Feature map before pooling (for skip connections)
- `P` (torch.Tensor): Pooled feature map (downsampled by factor of 2)

**Key Features**:
- **Feature Extraction**: Uses CNNBlock for feature learning
- **Downsampling**: MaxPool2d reduces spatial dimensions by 50%
- **Skip Connection Support**: Returns both processed and pooled features
- **Information Preservation**: Skip connections help preserve fine details

**Dimension Changes**:
```
Input:  (B, in_channels, H, W)
X_old:  (B, out_channels, H, W)      # Same spatial size
P:      (B, out_channels, H/2, W/2)  # Downsampled
```

**Usage Example**:
```python
encoder = EncoderBlock(in_channels=1, out_channels=64)
skip_features, downsampled = encoder(ct_image)
# skip_features: (B, 64, 362, 362) for skip connection
# downsampled: (B, 64, 181, 181) for next encoder level
```

---

#### DecoderBlock
**Purpose**: Decoder component for U-Net architecture with convolution and upsampling.

**Class Definition**:
```python
class DecoderBlock(nn.Module)
```

**Constructor Parameters**:
- `in_channels` (int): Number of input channels (from concatenated skip + upsampled)
- `out_channels1` (int): Intermediate channel count after convolution
- `out_channels2` (int): Final channel count after upsampling
- `output_padding` (int or tuple, default=0): Additional padding for transpose convolution

**Architecture**:
```
Input → CNNBlock → ConvTranspose2d(2×2, stride=2) → Output
```

**Key Features**:
- **Feature Processing**: CNNBlock processes concatenated features
- **Upsampling**: ConvTranspose2d doubles spatial dimensions
- **Flexible Padding**: output_padding handles dimension mismatches
- **Channel Reduction**: Typically reduces channel count while increasing resolution

**Dimension Changes**:
```
Input:  (B, in_channels, H, W)
After CNNBlock: (B, out_channels1, H, W)
Output: (B, out_channels2, 2*H, 2*W)  # Doubled spatial dimensions
```

**Usage Example**:
```python
# Standard decoder block
decoder = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)

# Handle dimension mismatch with output_padding
decoder_adjusted = DecoderBlock(
    in_channels=512, out_channels1=256, out_channels2=128, 
    output_padding=(1,1)  # Adds 1 pixel to each dimension
)
```

---

### 3. U-Net Architectures

#### UNet
**Purpose**: Standard U-Net implementation for medical image segmentation and reconstruction.

**Class Definition**:
```python
class UNet(nn.Module)
```

**Input/Output Specification**:
- **Input**: `(Batch, 1, 362, 362)` - CT sinogram or image
- **Output**: `(Batch, 1, 362, 362)` - Reconstructed image

**Architecture Overview**:
```
Encoder Path:    Input → E1 → E2 → E3 → E4 → Bridge
                   ↓     ↓     ↓     ↓
Decoder Path:           D1 ← D2 ← D3 ← D4 ← Bridge
                         ↓
                      Output
```

**Detailed Architecture**:

##### Encoder Path (Contracting):
1. **EncoderBlock1**: 1 → 64 channels, 362×362 → 181×181
2. **EncoderBlock2**: 64 → 128 channels, 181×181 → 90×90  
3. **EncoderBlock3**: 128 → 256 channels, 90×90 → 45×45
4. **EncoderBlock4**: 256 → 512 channels, 45×45 → 22×22

##### Bridge (Bottleneck):
- **CNNBlock**: 512 → 1024 channels
- **ConvTranspose2d**: 1024 → 512 channels, 22×22 → 45×45

##### Decoder Path (Expanding):
1. **DecoderBlock1**: 1024 → 256 channels, 45×45 → 90×90
2. **DecoderBlock2**: 512 → 128 channels, 90×90 → 181×181  
3. **DecoderBlock3**: 256 → 64 channels, 181×181 → 362×362

##### Output Layers:
- **Conv2d**: 128 → 64 channels (1×1 kernel)
- **Conv2d**: 64 → 1 channel (1×1 kernel)  
- **Sigmoid**: Output activation for [0,1] range

**Skip Connections**:
```python
# Concatenation pattern in forward pass
X = torch.cat([X_old4, X], dim=1)  # 512 + 512 = 1024 channels
X = torch.cat([X_old3, X], dim=1)  # 256 + 256 = 512 channels  
X = torch.cat([X_old2, X], dim=1)  # 128 + 128 = 256 channels
X = torch.cat([X_old1, X], dim=1)  # 64 + 64 = 128 channels
```

**Key Features**:
- **Skip Connections**: Preserve fine-grained details
- **Symmetric Architecture**: Balanced encoder-decoder structure
- **Sigmoid Activation**: Ensures output in [0,1] range
- **Medical Imaging Optimized**: Designed for 362×362 CT images

**Usage Example**:
```python
# Initialize model
unet = UNet(in_channels=1)

# Forward pass
ct_sinogram = torch.randn(8, 1, 362, 362)  # Batch of sinograms
reconstruction = unet(ct_sinogram)  # Reconstructed images
print(f"Output shape: {reconstruction.shape}")  # (8, 1, 362, 362)
print(f"Output range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")  # [0, 1]
```

---

#### UNet_pre
**Purpose**: U-Net variant with different output padding configuration for specific input dimensions.

**Key Differences from Standard UNet**:
- **Bridge Output Padding**: `(1,0)` instead of `(1,1)`
- **DecoderBlock3 Padding**: `(0,1)` for dimension adjustment
- **Use Case**: Handles slight dimensional mismatches in certain datasets

**Architecture Variations**:
```python
# Bridge layer difference
self.bridge = nn.Sequential(
    CNNBlock(in_channels=512, out_channels=1024),
    nn.ConvTranspose2d(in_channels=1024, out_channels=512, 
                      kernel_size=2, stride=2, output_padding=(1,0))  # Asymmetric padding
)

# Decoder block with custom padding
self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, 
                                  out_channels2=64, output_padding=(0,1))
```

**Usage Scenario**:
```python
# For specific preprocessing or non-standard input sizes
unet_pre = UNet_pre(in_channels=1)
custom_input = torch.randn(4, 1, 360, 364)  # Slightly different dimensions
output = unet_pre(custom_input)
```

---

#### UNet_no_activation & UNet_pre_no_activation
**Purpose**: U-Net variants without final sigmoid activation for unrestricted output ranges.

**Key Differences**:
- **No Sigmoid**: Allows negative values and values > 1
- **Raw Output**: Direct model output without range constraints
- **Use Cases**: 
  - Training with different loss functions
  - Post-processing with custom activation
  - Intermediate feature extraction

**Comparison**:
```python
# Standard UNet
unet_standard = UNet(in_channels=1)
output_standard = unet_standard(input)  # Range: [0, 1]

# No activation variant
unet_raw = UNet_no_activation(in_channels=1)
output_raw = unet_raw(input)  # Range: (-∞, +∞)

# Apply custom activation later
output_custom = torch.tanh(output_raw)  # Range: [-1, 1]
```

---

### 4. Specialized Architectures

#### PseudoResnet
**Purpose**: ResNet-inspired architecture with skip connections for image-to-image tasks.

**Class Definition**:
```python
class PseudoResnet(nn.Module)
```

**Architecture Design**:
```
Input (1 channel) → 3 → 9 → 27 → 81 → 81 → 27 → 9 → 3 → 1 Output
  ↓                                            ↑     ↑    ↑
  └─────────── Skip Connections ──────────────┘     │    │
                                                     │    │
              X3 ←─────────────────────────────────┘    │
              X2 ←──────────────────────────────────────┘
```

**Forward Pass Logic**:
```python
def forward(self, X):
    X1 = self.conv1(X)      # 1 → 3 channels
    X2 = self.conv2(X1)     # 3 → 9 channels  
    X3 = self.conv3(X2)     # 9 → 27 channels
    X4 = self.conv4(X3)     # 27 → 81 channels
    
    X5 = self.conv5(X4)     # 81 → 81 channels (bottleneck)
    
    X6 = self.conv6(X5 + X4)  # Skip connection: 81 + 81 → 27
    X7 = self.conv7(X6 + X3)  # Skip connection: 27 + 27 → 9  
    X8 = self.conv8(X7 + X2)  # Skip connection: 9 + 9 → 3
    
    return X8 + X1           # Final skip: 3 + 1 (broadcasted)
```

**Key Features**:
- **Symmetric Channel Progression**: 1→3→9→27→81→27→9→3→1
- **Multiple Skip Connections**: Facilitates gradient flow and feature reuse
- **Residual Learning**: Learns residual mappings rather than direct mappings
- **Compact Architecture**: Fewer parameters than traditional ResNet

**Channel Broadcasting Note**:
The final addition `X8 + X1` relies on PyTorch's broadcasting:
- X8 shape: (B, 3, H, W)  
- X1 shape: (B, 1, H, W)
- Result: (B, 3, H, W) with X1 broadcasted across the 3 channels

**Code Issue Identification**:
```python
# Potential bug in the original code:
self.conv6 = CNNBlock(in_channels=81, out_channels=27)
self.conv6 = CNNBlock(in_channels=27, out_channels=9)  # Overwrites previous line
self.conv7 = CNNBlock(in_channels=9, out_channels=3)
```

**Corrected Implementation**:
```python
self.conv6 = CNNBlock(in_channels=81, out_channels=27)
self.conv7 = CNNBlock(in_channels=27, out_channels=9)    # Should be conv7
self.conv8 = CNNBlock(in_channels=9, out_channels=3)     # Should be conv8
self.conv9 = CNNBlock(in_channels=3, out_channels=1)     # Should be conv9
```

**Usage Example**:
```python
pseudo_resnet = PseudoResnet()
ct_image = torch.randn(4, 1, 256, 256)
enhanced_image = pseudo_resnet(ct_image)
print(f"Input shape: {ct_image.shape}")           # (4, 1, 256, 256)
print(f"Output shape: {enhanced_image.shape}")    # (4, 1, 256, 256)
```

---

### 5. Frequency Domain Components

#### TrainableFourierSeries
**Purpose**: Learnable frequency domain filter based on Fourier series representation.

**Class Definition**:
```python
class TrainableFourierSeries(nn.Module)
```

**Constructor Parameters**:
- `freqs` (torch.Tensor): Frequency values for filter definition
- `init_filter` (torch.Tensor): Initial filter values for coefficient extraction
- `L` (int, default=50): Number of Fourier series terms

**Mathematical Foundation**:
A Fourier series represents a periodic function as:
```
f(x) = a₀ + Σ(aᵢcos(2πix) + bᵢsin(2πix))
```

**Implementation Process**:

##### 1. Coefficient Extraction (`cos_sin_coeffs`):
```python
def cos_sin_coeffs(self, f, L):
    fft_result = torch.fft.fft(f)
    a0 = fft_result[0].real / N                    # DC component
    
    for i in range(1, L + 1):
        ai = (2 / N) * fft_result[i].real          # Cosine coefficients
        bi = (-2 / N) * fft_result[i].imag         # Sine coefficients
```

##### 2. Frequency Normalization:
```python
freqs_range = freqs.max() - freqs.min()
normalized_freqs = (freqs - freqs.min()) / freqs_range
```

##### 3. Basis Function Matrix:
```python
cos_terms = torch.cos(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)
sin_terms = torch.sin(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)
cos_sin_stuff = torch.cat([cos_terms, sin_terms], dim=1)  # Shape: (freq_len, 2*L)
```

##### 4. Filter Reconstruction:
```python
filter = self.const + torch.matmul(self.cos_sin_stuff, self.coeffs)
filter = torch.fft.fftshift(filter)  # Center DC component
return filter.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,1,513)
```

**Learnable Parameters**:
- `self.coeffs`: Fourier coefficients (cosine and sine terms)
- `self.const`: DC component (a₀)

**Key Features**:
- **Smooth Filters**: Fourier basis ensures smooth frequency responses
- **Parametric Efficiency**: Few parameters (2*L + 1) represent complex filters
- **Differentiable**: All operations support gradient computation
- **Flexible Initialization**: Can start from any initial filter design

**Usage Example**:
```python
# Initialize with frequency grid and initial filter
freqs = torch.linspace(0, 1, 513)  # Normalized frequencies
initial_filter = torch.ones(513)   # Start with all-pass filter

fourier_filter = TrainableFourierSeries(freqs, initial_filter, L=25)

# Generate learnable filter
dummy_input = None  # Not used in forward pass
filter_response = fourier_filter(dummy_input)
print(f"Filter shape: {filter_response.shape}")  # (1, 1, 1, 513)

# Apply to frequency domain data
fft_data = torch.fft.fft(input_signal)
filtered_data = fft_data * filter_response.squeeze()
```

**Applications**:
- **CT Reconstruction**: Frequency domain filtering for noise reduction
- **Signal Processing**: Adaptive filter design
- **Image Enhancement**: Learnable frequency emphasis/suppression

---

#### LearnableWindow
**Purpose**: Simple learnable weighting function for 1D frequency domain filtering.

**Class Definition**:
```python
class LearnableWindow(nn.Module)
```

**Constructor Parameters**:
- `init_tensor` (torch.Tensor, default=torch.ones(513)): Initial window values

**Architecture**:
```python
self.weights = nn.Parameter(init_tensor)  # Learnable weights

def forward(self, x):
    return self.weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,513)
```

**Key Features**:
- **Direct Parameterization**: Each frequency bin has independent weight
- **Maximum Flexibility**: No smoothness constraints
- **Simple Implementation**: Direct parameter optimization
- **Initialization Control**: Can start with specific window shapes

**Usage Example**:
```python
# Initialize with Hamming window
hamming_window = torch.hamming_window(513)
learnable_window = LearnableWindow(init_tensor=hamming_window)

# Apply to frequency data
freq_data = torch.randn(4, 1, 1, 513)  # Batch of frequency data
windowed_data = freq_data * learnable_window(None)
```

---

#### LearnableWindowII
**Purpose**: 2D learnable weighting function for full sinogram frequency domain filtering.

**Class Definition**:
```python
class LearnableWindowII(nn.Module)
```

**Constructor Parameters**:
- `init_tensor` (torch.Tensor, default=torch.ones((1000,513))): Initial 2D window

**Key Features**:
- **2D Filtering**: Operates on both angle and detector dimensions
- **Full Sinogram Coverage**: 1000 angles × 513 detector elements
- **Independent Weights**: Each sinogram element has learnable weight
- **Memory Intensive**: Large parameter count (1000 × 513 = 513,000 parameters)

**Dimension Explanation**:
- **1000**: Number of projection angles in CT acquisition
- **513**: Number of detector elements per projection
- **Output**: (1, 1, 1000, 513) for broadcasting with sinogram batches

**Usage Example**:
```python
# Initialize with uniform weighting
uniform_weights = torch.ones((1000, 513))
learnable_2d_window = LearnableWindowII(init_tensor=uniform_weights)

# Apply to sinogram data
sinogram_batch = torch.randn(8, 1, 1000, 513)  # Batch of sinograms
weighted_sinograms = sinogram_batch * learnable_2d_window(None)

print(f"Parameter count: {sum(p.numel() for p in learnable_2d_window.parameters())}")
# Output: 513000 parameters
```

---

## Architecture Comparison and Selection Guide

### Model Complexity Comparison

| Model | Parameters | Use Case | Input/Output | Activation |
|-------|------------|----------|--------------|------------|
| UNet | ~31M | Standard reconstruction | (B,1,362,362) | Sigmoid |
| UNet_pre | ~31M | Custom dimensions | (B,1,H,W) | Sigmoid |
| UNet_no_activation | ~31M | Raw outputs | (B,1,362,362) | None |
| PseudoResnet | ~2M | Lightweight enhancement | (B,1,H,W) | None |
| TrainableFourierSeries | 2*L+1 | Frequency filtering | (1,1,1,513) | None |
| LearnableWindow | 513 | 1D frequency weighting | (1,1,1,513) | None |
| LearnableWindowII | 513K | 2D sinogram weighting | (1,1,1000,513) | None |

### Selection Criteria

#### For CT Reconstruction:
```python
# High-quality reconstruction with skip connections
model = UNet(in_channels=1)

# Memory-constrained environments  
model = PseudoResnet()

# Custom loss functions or post-processing
model = UNet_no_activation(in_channels=1)
```

#### For Frequency Domain Processing:
```python
# Smooth, parameterized filters
filter_model = TrainableFourierSeries(freqs, init_filter, L=50)

# Maximum flexibility, independent weights
filter_model = LearnableWindow(init_tensor=custom_window)

# Full sinogram processing
filter_model = LearnableWindowII(init_tensor=custom_2d_window)
```

---

## Implementation Best Practices

### 1. Memory Management
```python
def efficient_model_usage():
    # Use gradient checkpointing for large models
    model = UNet(in_channels=1)
    model.gradient_checkpointing = True
    
    # Clear cache between batches
    torch.cuda.empty_cache()
    
    # Use mixed precision training
    from torch.cuda.amp import autocast
    with autocast():
        output = model(input)
```

### 2. Dimension Handling
```python
def handle_dimension_mismatches():
    # Check input dimensions
    def validate_input(x, expected_shape):
        if x.shape[-2:] != expected_shape[-2:]:
            x = F.interpolate(x, size=expected_shape[-2:], mode='bilinear')
        return x
    
    # Flexible model wrapper
    class FlexibleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet(in_channels=1)
            
        def forward(self, x):
            original_size = x.shape[-2:]
            x = validate_input(x, (362, 362))
            x = self.unet(x)
            if original_size != (362, 362):
                x = F.interpolate(x, size=original_size, mode='bilinear')
            return x
```

### 3. Model Initialization
```python
def initialize_models():
    # Xavier initialization for Conv2d layers
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model = UNet(in_channels=1)
    model.apply(init_weights)
    
    # He initialization for ReLU networks
    def init_he_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    
    model.apply(init_he_weights)
```

---

## Advanced Usage Patterns

### 1. Multi-Scale Training
```python
class MultiScaleTraining:
    def __init__(self, model):
        self.model = model
        self.scales = [0.5, 0.75, 1.0, 1.25]
    
    def forward(self, x):
        outputs = []
        for scale in self.scales:
            if scale != 1.0:
                scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear')
                scaled_out = self.model(scaled_x)
                scaled_out = F.interpolate(scaled_out, size=x.shape[-2:], mode='bilinear')
            else:
                scaled_out = self.model(x)
            outputs.append(scaled_out)
        
        return torch.mean(torch.stack(outputs), dim=0)
```

### 2. Ensemble Methods
```python
class ModelEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0/len(models)] * len(models)
    
    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(weight * output)
        return torch.sum(torch.stack(outputs), dim=0)

# Usage
ensemble = ModelEnsemble([
    UNet(in_channels=1),
    UNet_pre(in_channels=1), 
    PseudoResnet()
], weights=[0.5, 0.3, 0.2])
```

### 3. Progressive Training
```python
class ProgressiveTrainer:
    def __init__(self, model):
        self.model = model
        self.current_resolution = 64
        self.target_resolution = 362
    
    def get_current_input_size(self):
        return (self.current_resolution, self.current_resolution)
    
    def increase_resolution(self):
        if self.current_resolution < self.target_resolution:
            self.current_resolution = min(self.current_resolution * 2, self.target_resolution)
    
    def prepare_input(self, x):
        current_size = self.get_current_input_size()
        return F.interpolate(x, size=current_size, mode='bilinear')
```

---

## Integration with Training Pipelines

### 1. Standard Training Loop
```python
def train_unet(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        scheduler.step(val_loss)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```

### 2. Model Saving and Loading
```python
def save_model(model, path, epoch, optimizer=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': {'in_channels': getattr(model, 'in_channels', 1)}
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)

def load_model(path, model_class_dict):
    checkpoint = torch.load(path)
    model_class = model_class_dict[checkpoint['model_class']]
    model = model_class(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']

# Usage
model_classes = {
    'UNet': UNet,
    'UNet_pre': UNet_pre,
    'PseudoResnet': PseudoResnet
}
model, epoch = load_model('checkpoint.pth', model_classes)
```

This comprehensive documentation covers all architectural components in the Modells.py file, providing both theoretical understanding and practical implementation guidance for medical image processing and CT reconstruction applications.
