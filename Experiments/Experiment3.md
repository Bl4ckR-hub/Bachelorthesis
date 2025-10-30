# Experiment 3: Documentation

## Overview
This script implements a distributed training experiment for post-processing CT reconstructions using a UNet architecture. The experiment uses a filtered backprojection (FBP) pipeline with a fixed filtering and backprojection module, and trains only the UNet post-processing module to improve image quality. Training is performed using PyTorch's DistributedDataParallel (DDP) for multi-GPU support.

## Main Components
- **Imports**: Utilizes PyTorch, custom modules (`CT_library`, `Reconstructor`, `Criterion`, `Trainer`, `Modells`), and `piq` for image quality metrics.
- **Distributed Training**: Employs PyTorch's DDP, multiprocessing, and distributed samplers for efficient multi-GPU training.
- **Directories**: Paths for data, checkpoints, and results are set at the top for easy modification.

## Key Functions
### setup & cleanup
- `setup(rank, world_size)`: Initializes the distributed process group for DDP.
- `cleanup()`: Destroys the process group after training.

### main(rank, world_size)
- Initializes distributed training environment.
- Loads training and validation datasets using distributed samplers.
- Constructs the FBP model with a learnable window and ramp filter, but only the UNet post-processing module is trainable.
- Freezes parameters of the filtering and backprojection modules.
- Wraps the model in DDP for multi-GPU training.
- Sets up Adam optimizer for the UNet parameters and MSE loss for training.
- Initializes a learning rate scheduler.
- Uses a custom `Trainer` class for training and validation.
- Trains the UNet in 5-epoch increments, saving losses and metrics after each segment.
- Cleans up the distributed environment.

### Multiprocessing Entry Point
- Uses `mp.spawn` to launch training across all available GPUs.

## Model Architecture
- **Learnable WindowII**: Window function (parameters frozen).
- **Ramp Filter**: Standard filter for FBP (parameters frozen).
- **Filtering Module**: Combines window and filter (parameters frozen).
- **Vanilla Backprojection**: Standard backprojection operation (parameters frozen).
- **UNet**: Post-processing module, the only part being trained.
- **LearnableFBP**: Full FBP pipeline with UNet as post-processing.

## Training Details
- **Distributed Training**: Each GPU trains on a subset of the data using DDP.
- **Batch Size**: 4
- **Epochs**: 25 (5 epochs per segment, repeated 5 times)
- **Optimizer**: Adam (UNet parameters only)
- **Scheduler**: ReduceLROnPlateau
- **Checkpointing**: Saves best and latest model states.
- **Results**: Loss and metric histories saved to disk after each segment.

## Purpose
This experiment aims to improve CT image reconstructions by training a UNet as a post-processing step after a fixed FBP pipeline. Only the UNet is trained, allowing for focused enhancement of reconstruction quality.

## Customization
- Modify dataset paths, batch size, or training parameters at the top of the script.
- Adjust number of epochs or optimizer settings in the main function.

## Output
- Trained UNet model checkpoints (`best.pth`)
- Loss histories (`losses5.pth`, `losses10.pth`, etc.)
- Metric histories (`metrics5.pth`, `metrics10.pth`, etc.)

---
**File:** `Experiments/2/UNet.py`
**Experiment:** Distributed training of a UNet post-processing module for CT reconstruction using a fixed FBP pipeline.
