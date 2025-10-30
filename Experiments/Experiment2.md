# Experiment 2: Documentation

## Overview
This script implements a distributed training experiment for a learnable window function in a filtered backprojection (FBP) CT reconstruction pipeline. The experiment uses a combination of perceptual, edge, and L1 losses, and leverages PyTorch's DistributedDataParallel (DDP) for multi-GPU training on a subset of the LoDoPaB dataset.

## Main Components
- **Imports**: Uses PyTorch, custom modules (`CT_library`, `Reconstructor`, `Criterion`, `Trainer`, `Modells`, `metrics`), and `piq` for perceptual metrics.
- **Distributed Training**: Utilizes PyTorch's DDP, multiprocessing, and distributed samplers for efficient multi-GPU training.
- **Directories**: Paths for data, checkpoints, and results are set at the top for easy modification.

## Key Functions
### setup & cleanup
- `setup(rank, world_size)`: Initializes the distributed process group for DDP.
- `cleanup()`: Destroys the process group after training.

### main(rank, world_size)
- Initializes distributed training environment.
- Loads training and validation datasets using distributed samplers.
- Constructs the FBP model with a learnable window and ramp filter.
- Loads pre-trained model and optimizer states if available.
- Defines a composite loss function: weighted sum of L1, MultiScale SSIM, and Gradient Edge Loss.
- Sets up a learning rate scheduler.
- Initializes the custom `Trainer` class for training and validation.
- Trains the model for 10 epochs and saves the loss history.
- Cleans up the distributed environment.

### Multiprocessing Entry Point
- Uses `mp.spawn` to launch training across all available GPUs.

## Loss Function
The loss function combines:
- **L1 Loss**: Measures pixel-wise differences.
- **MultiScale SSIM Loss**: Measures perceptual similarity.
- **Gradient Edge Loss**: Encourages edge preservation.

## Model Architecture
- **Learnable WindowII**: Window function with trainable parameters.
- **Ramp Filter**: Standard filter for FBP.
- **Filtering Module**: Combines window and filter.
- **Vanilla Backprojection**: Standard backprojection operation.
- **LearnableFBP**: Full FBP pipeline with learnable window.

## Training Details
- **Distributed Training**: Each GPU trains on a subset of the data using DDP.
- **Batch Size**: 4
- **Epochs**: 10
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
- **Checkpointing**: Saves best and latest model states.
- **Results**: Loss history saved to disk.

## Purpose
This experiment aims to train a learnable window in the FBP pipeline using a combination of perceptual and edge-aware losses, leveraging distributed training for scalability. It builds on previous experiments by introducing more complex loss functions and multi-GPU support.

## Customization
- Modify dataset paths, batch size, or training parameters at the top of the script.
- Adjust loss weights or number of epochs in the main function.

## Output
- Trained model checkpoints (`best.pth`)
- Loss history (`losses20.pth`)

---
**File:** `Experiments/1/windowII+edge+perceptual+subset.py`
**Experiment:** Distributed training of a learnable window in FBP using perceptual and edge-aware losses on a CT image subset.
