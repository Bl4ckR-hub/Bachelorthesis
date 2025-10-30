# Experiment 1: Documentation

## Overview
This script performs distributed training of a learnable window function within a filtered backprojection (FBP) pipeline for CT image reconstruction. The experiment uses the LoDoPaB dataset and optimizes the window parameters using MSE loss, leveraging PyTorch's DistributedDataParallel (DDP) for multi-GPU scalability.

## Main Components
- **Imports**: PyTorch, custom modules (`CT_library`, `Reconstructor`, `Criterion`, `Trainer`, `Modells`), and `piq` for metrics.
- **Distributed Training**: Uses DDP, multiprocessing, and distributed samplers for efficient multi-GPU training.
- **Directories**: Paths for data, checkpoints, and results are set at the top for easy modification.

## Key Functions
### setup & cleanup
- `setup(rank, world_size)`: Initializes the distributed process group for DDP.
- `cleanup()`: Destroys the process group after training.

### main(rank, world_size)
- Initializes distributed training environment.
- Loads training and validation datasets using distributed samplers.
- Constructs the FBP model with a learnable window and ramp filter.
- Uses an identity post-processing module (no additional processing).
- Wraps the model in DDP for multi-GPU training.
- Sets up Adam optimizer for the window parameters and MSE loss for training.
- Initializes a learning rate scheduler and early stopping.
- Uses a custom `Trainer` class for training and validation.
- Trains the model for 30 epochs and saves losses and metrics after training.
- Cleans up the distributed environment.

### Multiprocessing Entry Point
- Uses `mp.spawn` to launch training across all available GPUs.

## Model Architecture
- **Learnable WindowII**: Window function with trainable parameters.
- **Ramp Filter**: Standard filter for FBP.
- **Filtering Module**: Combines window and filter.
- **Vanilla Backprojection**: Standard backprojection operation.
- **Identity**: No post-processing applied.
- **LearnableFBP**: Full FBP pipeline with learnable window.

## Training Details
- **Distributed Training**: Each GPU trains on a subset of the data using DDP.
- **Batch Size**: 4
- **Epochs**: 30
- **Optimizer**: Adam (window parameters only)
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience set to 4 epochs without improvement.
- **Checkpointing**: Saves best and latest model states.
- **Results**: Loss and metric histories saved to disk after training.

## Purpose
This experiment aims to train a learnable window in the FBP pipeline using MSE loss, leveraging distributed training for scalability. It serves as a baseline for window optimization without additional post-processing.

## Customization
- Modify dataset paths, batch size, or training parameters at the top of the script.
- Adjust number of epochs or optimizer settings in the main function.

## Output
- Trained window model checkpoints (`best.pth`)
- Loss history (`losses.pth`)
- Metric history (`metrics.pth`)

---
**File:** `Experiments/1/windowII+l2+slow_subset.py`
**Experiment:** Distributed training of a learnable window in FBP using MSE loss on a CT image subset.
