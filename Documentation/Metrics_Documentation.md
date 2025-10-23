# Metrics.py - Comprehensive Documentation

## Overview

This module provides a collection of essential evaluation metrics for image quality assessment, specifically designed for medical image processing and CT reconstruction tasks. The metrics implement standard image comparison functions with proper batch handling and numerical stability considerations.

## Dependencies

```python
import torch
import torch.nn.functional as F
import piq  # PyTorch Image Quality Assessment library
```

### Installation Requirements
```bash
pip install torch torchvision
pip install piq  # For advanced image quality metrics
```

## Metric Categories

### 1. Basic Loss Functions

#### l1_loss
**Purpose**: Computes the L1 (Mean Absolute Error) loss between predicted and target images.

**Function Signature**:
```python
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

**Parameters**:
- `pred` (torch.Tensor): Predicted image tensor of any shape
- `target` (torch.Tensor): Ground truth image tensor (same shape as pred)

**Returns**:
- `torch.Tensor`: Scalar tensor containing the mean L1 loss

**Mathematical Formula**:
```
L1 = (1/N) * Σ|pred - target|
```
where N is the total number of elements across all dimensions.

**Characteristics**:
- **Reduction**: `'mean'` - averages across batch and all spatial dimensions
- **Robustness**: Less sensitive to outliers compared to L2 loss
- **Gradient Properties**: Non-differentiable at zero, but subgradient exists
- **Use Cases**: Edge preservation, robust regression, sparse solutions

**Usage Example**:
```python
pred = torch.randn(8, 1, 256, 256)    # Batch of 8 CT images
target = torch.randn(8, 1, 256, 256)  # Ground truth images
loss = l1_loss(pred, target)
print(f"L1 Loss: {loss.item():.6f}")
```

---

#### l2_loss
**Purpose**: Computes the L2 (Sum of Squared Errors) loss between predicted and target images.

**Function Signature**:
```python
def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

**Parameters**:
- `pred` (torch.Tensor): Predicted image tensor
- `target` (torch.Tensor): Ground truth image tensor

**Returns**:
- `torch.Tensor`: Scalar tensor containing the sum of squared differences

**Mathematical Formula**:
```
L2 = Σ(pred - target)²
```

**Characteristics**:
- **Reduction**: `'sum'` - sums across all elements without averaging
- **Sensitivity**: Heavily penalizes large errors (quadratic penalty)
- **Gradient Properties**: Smooth gradients, easier optimization
- **Scale Dependency**: Result scales with tensor size

**Important Note**: 
This function returns the sum rather than mean, making it sensitive to batch size and image dimensions. Consider using `mse_loss` for normalized results.

**Usage Example**:
```python
pred = torch.randn(4, 1, 128, 128)
target = torch.randn(4, 1, 128, 128)
loss = l2_loss(pred, target)
print(f"L2 Loss (sum): {loss.item():.2f}")
```

---

#### mse_loss
**Purpose**: Computes the Mean Squared Error (MSE) loss between predicted and target images.

**Function Signature**:
```python
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

**Parameters**:
- `pred` (torch.Tensor): Predicted image tensor
- `target` (torch.Tensor): Ground truth image tensor

**Returns**:
- `torch.Tensor`: Scalar tensor containing the mean squared error

**Mathematical Formula**:
```
MSE = (1/N) * Σ(pred - target)²
```

**Characteristics**:
- **Reduction**: `'mean'` - averages across all elements
- **Scale Independence**: Normalized by total number of elements
- **Standard Metric**: Most commonly used regression loss
- **Relationship**: MSE = L2_loss / N, where N is total elements

**Usage Example**:
```python
pred = torch.randn(8, 1, 362, 362)
target = torch.randn(8, 1, 362, 362)
mse = mse_loss(pred, target)
print(f"MSE: {mse.item():.6f}")

# Verify relationship with l2_loss
l2 = l2_loss(pred, target)
n_elements = pred.numel()
assert torch.allclose(mse, l2 / n_elements)
```

---

### 2. Image Quality Metrics

#### psnr
**Purpose**: Computes Peak Signal-to-Noise Ratio, a standard metric for image quality assessment.

**Function Signature**:
```python
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor
```

**Parameters**:
- `pred` (torch.Tensor): Predicted image tensor of shape (B, C, H, W)
- `target` (torch.Tensor): Ground truth image tensor (same shape as pred)
- `max_val` (float, default=1.0): Maximum possible pixel value in the images

**Returns**:
- `torch.Tensor`: Scalar tensor containing the average PSNR across the batch

**Mathematical Formula**:
```
PSNR = 20 * log₁₀(MAX_VAL / √MSE)
```
where:
- MAX_VAL is the maximum possible pixel value
- MSE is the mean squared error per sample

**Implementation Details**:
1. **Per-Sample Calculation**: Computes PSNR for each sample in the batch individually
2. **Batch Averaging**: Returns the mean PSNR across all samples
3. **Numerical Stability**: Adds epsilon (1e-8) to prevent log(0)
4. **Dimension Handling**: Flattens spatial and channel dimensions per sample

**PSNR Value Interpretation**:
- **> 40 dB**: Excellent quality, imperceptible differences
- **30-40 dB**: Good quality, minor visible differences
- **20-30 dB**: Fair quality, noticeable differences
- **< 20 dB**: Poor quality, significant degradation

**Usage Examples**:
```python
# Example 1: Normalized images (0-1 range)
pred = torch.rand(4, 1, 256, 256)     # Random predictions
target = torch.rand(4, 1, 256, 256)   # Random targets
psnr_val = psnr(pred, target, max_val=1.0)
print(f"PSNR: {psnr_val.item():.2f} dB")

# Example 2: 8-bit images (0-255 range)
pred_8bit = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.float32)
target_8bit = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.float32)
psnr_val = psnr(pred_8bit, target_8bit, max_val=255.0)
print(f"PSNR (8-bit): {psnr_val.item():.2f} dB")

# Example 3: Perfect reconstruction
identical = torch.randn(1, 1, 64, 64)
psnr_perfect = psnr(identical, identical.clone())
print(f"Perfect PSNR: {psnr_perfect.item():.1f} dB")  # Should be very high
```

**Common max_val Values**:
- `1.0`: For normalized images in [0, 1] range
- `255.0`: For 8-bit images in [0, 255] range
- `65535.0`: For 16-bit images
- Custom values for specific data ranges

---

#### ssim_metric
**Purpose**: Computes Structural Similarity Index Measure (SSIM), which considers luminance, contrast, and structure.

**Function Signature**:
```python
def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

**Parameters**:
- `pred` (torch.Tensor): Predicted image tensor of shape (B, C, H, W)
- `target` (torch.Tensor): Ground truth image tensor (same shape as pred)

**Returns**:
- `torch.Tensor`: Scalar tensor containing the average SSIM across the batch

**Mathematical Foundation**:
SSIM combines three components:

1. **Luminance**: 
   ```
   l(x,y) = (2μₓμᵧ + c₁) / (μₓ² + μᵧ² + c₁)
   ```

2. **Contrast**: 
   ```
   c(x,y) = (2σₓσᵧ + c₂) / (σₓ² + σᵧ² + c₂)
   ```

3. **Structure**: 
   ```
   s(x,y) = (σₓᵧ + c₃) / (σₓσᵧ + c₃)
   ```

**Final SSIM**:
```
SSIM(x,y) = l(x,y) * c(x,y) * s(x,y)
```

**Implementation Details**:
- **Library**: Uses `piq.ssim` for robust, optimized implementation
- **Data Range**: Fixed at 1.0 (assumes normalized images)
- **Per-Sample Processing**: Computes SSIM per sample, then averages
- **Window**: Uses default Gaussian window (typically 11×11)

**SSIM Value Interpretation**:
- **1.0**: Perfect structural similarity (identical images)
- **0.9-1.0**: Excellent quality, very high similarity
- **0.7-0.9**: Good quality, acceptable similarity
- **0.5-0.7**: Fair quality, noticeable differences
- **< 0.5**: Poor quality, significant structural differences

**Advantages of SSIM**:
- **Perceptual Relevance**: Better correlation with human visual perception
- **Multi-Component**: Considers luminance, contrast, and structure
- **Robustness**: Less sensitive to uniform brightness/contrast changes
- **Range**: Bounded between -1 and 1 (typically 0 to 1 for natural images)

**Usage Examples**:
```python
# Example 1: High-quality reconstruction
pred = torch.randn(4, 1, 256, 256)
target = pred + 0.01 * torch.randn_like(pred)  # Add small noise
ssim_val = ssim_metric(pred, target)
print(f"SSIM: {ssim_val.item():.4f}")  # Should be close to 1.0

# Example 2: Comparing with MSE
pred = torch.rand(8, 1, 128, 128)
target = torch.rand(8, 1, 128, 128)
ssim_val = ssim_metric(pred, target)
mse_val = mse_loss(pred, target)
print(f"SSIM: {ssim_val.item():.4f}, MSE: {mse_val.item():.6f}")

# Example 3: Perfect match
identical = torch.randn(2, 1, 64, 64)
ssim_perfect = ssim_metric(identical, identical.clone())
print(f"Perfect SSIM: {ssim_perfect.item():.6f}")  # Should be 1.0
```

**Preprocessing Requirements**:
- Images should be normalized to [0, 1] range
- Minimum image size: typically 11×11 (for default window)
- Same dynamic range for both images

---

## Metric Comparison and Selection Guide

### Loss Functions vs. Quality Metrics

| Metric | Type | Range | Best For | Characteristics |
|--------|------|-------|----------|----------------|
| L1 Loss | Loss | [0, ∞) | Training, Edge preservation | Robust to outliers |
| L2 Loss | Loss | [0, ∞) | Training, Smooth gradients | Penalizes large errors |
| MSE | Loss | [0, ∞) | Training, General purpose | Normalized L2 |
| PSNR | Quality | [0, ∞) dB | Evaluation, Signal quality | Log scale, peak-based |
| SSIM | Quality | [-1, 1] | Evaluation, Perceptual quality | Structure-aware |

### Recommended Usage Patterns

#### For Training (Loss Functions):
```python
# Primary loss for reconstruction
primary_loss = mse_loss(pred, target)

# Alternative for edge-preserving tasks
edge_preserving_loss = l1_loss(pred, target)

# Combined loss
total_loss = 0.7 * mse_loss(pred, target) + 0.3 * l1_loss(pred, target)
```

#### For Evaluation (Quality Metrics):
```python
def evaluate_model(model, test_loader):
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    
    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(test_loader):
            pred = model(input_data)
            
            # Compute metrics
            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_metric(pred, target)
            batch_mse = mse_loss(pred, target)
            
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_mse += batch_mse.item()
    
    n_batches = len(test_loader)
    return {
        'PSNR': total_psnr / n_batches,
        'SSIM': total_ssim / n_batches,
        'MSE': total_mse / n_batches
    }
```

---

## Mathematical Properties and Relationships

### Error Metrics Relationships
```python
# Relationship between L2 and MSE
l2_val = l2_loss(pred, target)
mse_val = mse_loss(pred, target)
n_elements = pred.numel()
assert torch.allclose(mse_val, l2_val / n_elements)

# Relationship between MSE and PSNR
mse_val = mse_loss(pred, target)
psnr_val = psnr(pred, target, max_val=1.0)
# PSNR ≈ 20 * log10(1.0 / sqrt(mse_val))
```

### Metric Sensitivity Analysis
```python
def analyze_metric_sensitivity():
    # Create base image
    base = torch.ones(1, 1, 64, 64) * 0.5
    
    # Test different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    for noise_std in noise_levels:
        noisy = base + noise_std * torch.randn_like(base)
        
        l1_val = l1_loss(noisy, base).item()
        mse_val = mse_loss(noisy, base).item()
        psnr_val = psnr(noisy, base).item()
        ssim_val = ssim_metric(noisy, base).item()
        
        print(f"Noise σ={noise_std:.2f}: L1={l1_val:.4f}, "
              f"MSE={mse_val:.4f}, PSNR={psnr_val:.2f}dB, "
              f"SSIM={ssim_val:.4f}")
```

---

## Best Practices and Considerations

### 1. Numerical Stability
- **PSNR**: Includes epsilon (1e-8) to prevent log(0)
- **Division by Zero**: All metrics handle edge cases appropriately
- **Data Types**: Use float32 or float64 for precise calculations

### 2. Batch Processing
- **Per-Sample Metrics**: PSNR and SSIM compute per-sample then average
- **Memory Efficiency**: Metrics handle large batches efficiently
- **Consistent Shapes**: Ensure pred and target have identical shapes

### 3. Data Range Considerations
```python
# Ensure proper data range for each metric
def prepare_for_metrics(images):
    # Normalize to [0, 1] for SSIM and PSNR
    images = torch.clamp(images, 0, 1)
    return images

# Example usage
pred_norm = prepare_for_metrics(pred)
target_norm = prepare_for_metrics(target)
ssim_val = ssim_metric(pred_norm, target_norm)
```

### 4. Metric Aggregation
```python
class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {'l1': [], 'mse': [], 'psnr': [], 'ssim': []}
    
    def update(self, pred, target):
        self.metrics['l1'].append(l1_loss(pred, target).item())
        self.metrics['mse'].append(mse_loss(pred, target).item())
        self.metrics['psnr'].append(psnr(pred, target).item())
        self.metrics['ssim'].append(ssim_metric(pred, target).item())
    
    def compute_averages(self):
        return {k: sum(v) / len(v) for k, v in self.metrics.items()}
```

---

## Integration with Training Loops

### Simple Training Example
```python
def train_epoch(model, train_loader, optimizer, criterion=mse_loss):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader):
    model.eval()
    metrics = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            metrics['mse'] += mse_loss(output, target).item()
            metrics['psnr'] += psnr(output, target).item()
            metrics['ssim'] += ssim_metric(output, target).item()
    
    # Average over batches
    for key in metrics:
        metrics[key] /= len(val_loader)
    
    return metrics
```

---

## Advanced Usage and Extensions

### Custom Metric Combinations
```python
def weighted_quality_score(pred, target, weights={'psnr': 0.4, 'ssim': 0.6}):
    """Combine multiple metrics into a single quality score."""
    psnr_val = psnr(pred, target) / 40.0  # Normalize to ~[0,1]
    ssim_val = ssim_metric(pred, target)  # Already in [0,1]
    
    score = weights['psnr'] * psnr_val + weights['ssim'] * ssim_val
    return score

# Usage
quality = weighted_quality_score(pred, target)
print(f"Combined Quality Score: {quality.item():.4f}")
```

### Per-Channel Metrics
```python
def per_channel_metrics(pred, target):
    """Compute metrics for each channel separately."""
    assert pred.shape[1] == target.shape[1], "Channel mismatch"
    
    results = {}
    for c in range(pred.shape[1]):
        pred_c = pred[:, c:c+1, :, :]
        target_c = target[:, c:c+1, :, :]
        
        results[f'channel_{c}'] = {
            'mse': mse_loss(pred_c, target_c).item(),
            'psnr': psnr(pred_c, target_c).item(),
            'ssim': ssim_metric(pred_c, target_c).item()
        }
    
    return results
```

This comprehensive documentation covers all aspects of the Metrics.py module, providing both theoretical understanding and practical implementation guidance for image quality assessment in medical imaging and CT reconstruction applications.
