# Experiment 0: Documentation

## Overview
This script implements an experiment for training a learnable window function within a filtered backprojection (FBP) pipeline for computed tomography (CT) image reconstruction. The experiment focuses on overfitting a single image from the LoDoPaB dataset using a learnable window and a ramp filter, optimizing the window parameters with MSE loss.

## Main Components
- **Imports**: Uses PyTorch, custom modules (`Modells`, `Criterion`, `CT_library`, `Reconstructor`, `Metrics`), and `piq` for image quality metrics.
- **Directories**: Paths for training/validation data, model checkpoints, and results are set at the top for easy modification.
- **Device Selection**: Chooses GPU if available, otherwise CPU.

## Key Functions
### overfitter_preprocessing
Trains the model on a single image for a specified number of epochs. Implements early stopping and learning rate reduction if the loss plateaus. Saves the best window model parameters during training.
- **Inputs**: X (sinogram), Y (ground truth), model, optimizer, criterion, epochs, etc.
- **Logic**:
  - Moves data/model to device
  - Tracks loss and best parameters
  - Reduces learning rate if no improvement
  - Stops training if learning rate is too low
  - Saves losses for analysis

### evaluator
Evaluates the trained model on the input image using several metrics:
- PSNR
- L1 loss
- MSE loss
- SSIM

### save_params
Utility to save model parameters to disk.

## Main Experiment Flow
1. **Loss Function**: Uses MSE loss (sum reduction).
2. **Dataset**: Loads LoDoPaB dataset, selects one image (index 50).
3. **Model Construction**:
   - Learnable window (`LearnableWindowII`)
   - Ramp filter
   - Filtering module (combines window and filter)
   - Vanilla backprojection
   - Learnable FBP (combines all modules)
4. **Optimizer**: Adam optimizer for window parameters.
5. **Training**: Calls `overfitter_preprocessing` for 3000 epochs.
6. **Results**: Saves loss history to disk.

## Purpose
This experiment is designed to test the learnability and optimization of the window function in the FBP pipeline by overfitting a single image. It provides insights into the behavior of the window under ideal conditions and serves as a baseline for further experiments.

## Customization
- Change dataset paths, checkpoint locations, or training parameters at the top of the script.
- Adjust the number of epochs, learning rate, or early stopping criteria in the function arguments.

## Output
- Trained window model parameters (`params0.pth`)
- Loss history (`losses.pth`)

---
**File:** `Experiments/0/windowII+l2+slower.py`
**Experiment:** Overfitting a learnable window in FBP using MSE loss on a single CT image.
