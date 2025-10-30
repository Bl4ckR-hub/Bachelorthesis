# Experiment 5: Documentation

## Overview
This script implements distributed training of a UNet post-processing module for CT image reconstruction, using a composite perceptual loss applied in both the image and sinogram domains. The experiment uses a filtered backprojection (FBP) pipeline with fixed filtering and backprojection modules, and trains only the UNet to enhance reconstruction quality. Training is performed using PyTorch's DistributedDataParallel (DDP) for multi-GPU support.

## Main Difference to Experiment 4
**Key distinction:** In this experiment, the same composite loss function is applied to both the image domain and the sinogram domain (via the `criterion` and `criterion_sinogram` arguments in the trainer). This dual-domain loss encourages the UNet to improve not only the reconstructed images but also the corresponding sinograms, potentially leading to better overall consistency and reconstruction quality. In Experiment 4, the loss was applied only in the image domain.

## Main Components
- **Imports**: PyTorch, custom modules (`CT_library`, `Reconstructor`, `Criterion`, `Trainer`, `Modells`), and `piq` for perceptual metrics.
- **Distributed Training**: Uses DDP, multiprocessing, and distributed samplers for efficient multi-GPU training.
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
- Sets up Adam optimizer for the UNet parameters.
- Defines a composite loss function:
  - **L1 Loss** (alpha): pixel-wise similarity
  - **MultiScale SSIM Loss** (beta): perceptual/structural similarity
  - **Gradient Edge Loss** (gamma): edge preservation
  - **VGG Perceptual Loss** (delta): feature similarity (small weight for medical data)
- **Loss is applied in both image and sinogram domains.**
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
This experiment aims to improve CT image reconstructions by training a UNet as a post-processing step after a fixed FBP pipeline, using a composite perceptual loss for both image and sinogram domains. The dual-domain loss is the main difference to Experiment 4.

## Customization
- Modify dataset paths, batch size, or training parameters at the top of the script.
- Adjust loss weights or number of epochs in the main function.

## Output
- Trained UNet model checkpoints (`best.pth`)
- Loss histories (`losses5.pth`, `losses10.pth`, etc.)
- Metric histories (`metrics5.pth`, `metrics10.pth`, etc.)

---
**File:** `Experiments/5/UNet_sinoLoss.py`
**Experiment:** Distributed training of a UNet post-processing module for CT reconstruction using a composite perceptual loss in both image and sinogram domains.
