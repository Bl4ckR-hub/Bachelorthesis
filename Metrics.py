import torch
import torch.nn.functional as F
import piq


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Reduces across batch and all dimensions
    return F.l1_loss(pred, target, reduction='mean')

def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction='sum')


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction='mean')


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    # Compute PSNR per sample, then average over batch
    batch_size = pred.shape[0]
    mse = F.mse_loss(pred, target, reduction='none')
    mse_per_sample = mse.reshape(batch_size, -1).mean(dim=1)
    psnr_per_sample = 20 * torch.log10(max_val / torch.sqrt(mse_per_sample + 1e-8))
    return psnr_per_sample.mean()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # piq.ssim returns a tensor of shape (batch_size,)
    ssim_per_sample = piq.ssim(pred, target, data_range=1.0, reduction='none')
    return ssim_per_sample.mean()
