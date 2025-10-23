import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import CT_library



class FFTBandLoss(nn.Module):
    """
    FFT-based loss that penalizes mismatches in a specific frequency band.
    """
    def __init__(self, low_freq_ratio=0.2, loss_fn=nn.L1Loss()):
        """
        Args:
            low_freq_ratio (float): The ratio of the low-frequency band to mask out. 
                                  A value of 0.2 means the central 20% of frequencies
                                  (low frequencies) are ignored, and the loss is
                                  calculated on the remaining 80% (high frequencies).
        """
        super(FFTBandLoss, self).__init__()
        self.low_freq_ratio = low_freq_ratio
        self.loss_fn = loss_fn

    def create_high_freq_mask(self, shape, device):
        """Creates a mask to isolate high frequencies."""
        b, c, h, w = shape
        mask = torch.ones((h, w), device=device)
        
        # Calculate the center and the radius for the low-frequency circle
        center_h, center_w = h // 2, w // 2
        radius = int(min(center_h, center_w) * self.low_freq_ratio)

        # Create a circular mask for the low-frequency components
        y, x = torch.ogrid[-center_h:h-center_h, -center_w:w-center_w]
        low_freq_mask = x*x + y*y <= radius*radius
        
        # Invert the mask to keep only high frequencies
        mask[low_freq_mask] = 0
        
        return mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

    def forward(self, pred, target):
        """
        Calculates the FFT Band Loss.
        
        Args:
            pred (torch.Tensor): The predicted image tensor (B, C, H, W).
            target (torch.Tensor): The ground truth image tensor (B, C, H, W).
            
        Returns:
            torch.Tensor: The calculated loss.
        """
        # --- Convert images to frequency domain ---
        # 1. Apply FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # 2. Shift the zero-frequency component to the center
        pred_fft_shifted = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft_shifted = torch.fft.fftshift(target_fft, dim=(-2, -1))
        
        # --- Create mask and apply it ---
        mask = self.create_high_freq_mask(pred.shape, pred.device)
        
        pred_high_freq = pred_fft_shifted * mask
        target_high_freq = target_fft_shifted * mask
        
        # --- Calculate loss on the magnitude of the high-frequency components ---
        loss = self.loss_fn(torch.abs(pred_high_freq), torch.abs(target_high_freq))
        
        return loss

# --- How to use it ---
# loss_fft = FFTBandLoss(low_freq_ratio=0.2)
# pred_image = torch.randn(4, 3, 256, 256)
# target_image = torch.randn(4, 3, 256, 256)
# loss = loss_fft(pred_image, target_image)
# loss.backward()

class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss, which encourages similarity in different frequency bands.
    """
    def __init__(self, max_levels=3, kernel_size=5, sigma=1.0, loss_fn=nn.L1Loss()):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.loss_fn = loss_fn
        
        # Create a Gaussian kernel for blurring
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)

    def create_gaussian_kernel(self, kernel_size, sigma):
        """Creates a 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2.0
        
        g = coords**2
        g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()
        
        g /= g.sum()
        return g.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, kernel_size, kernel_size)

    def laplacian_pyramid(self, img, levels):
        """Builds the Laplacian pyramid for an image."""
        pyramid = []
        current_img = img
        
        for level in range(levels):
            # Ensure kernel is on the same device and has the same dtype as the image
            kernel = self.kernel.repeat(img.shape[1], 1, 1, 1)
            
            # Blur the image
            blurred = F.conv2d(current_img, kernel, padding='same', groups=img.shape[1])
            
            # Downsample
            downsampled = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # Upsample
            upsampled = F.interpolate(downsampled, size=current_img.shape[2:], mode='bilinear', align_corners=False)
            
            # Subtract to get the Laplacian (high-frequency) layer
            laplacian_level = current_img - upsampled
            pyramid.append(laplacian_level)
            
            current_img = downsampled
            
        # The last level of the pyramid is the final downsampled image (low-frequency)
        pyramid.append(current_img)
        return pyramid

    def forward(self, pred, target):
        """
        Calculates the Laplacian Pyramid Loss.
        
        Args:
            pred (torch.Tensor): The predicted image tensor (B, C, H, W).
            target (torch.Tensor): The ground truth image tensor (B, C, H, W).
            
        Returns:
            torch.Tensor: The calculated loss.
        """
        pred_pyramid = self.laplacian_pyramid(pred, self.max_levels)
        target_pyramid = self.laplacian_pyramid(target, self.max_levels)
        
        total_loss = 0
        # Apply loss at each level of the pyramid
        # Weights can be adjusted, here we use 2^l to emphasize finer details
        for level in range(self.max_levels + 1):
            weight = 2.0 ** level
            total_loss += weight * self.loss_fn(pred_pyramid[level], target_pyramid[level])
            
        return total_loss / (self.max_levels + 1)

# --- How to use it ---
# loss_lap = LaplacianPyramidLoss(max_levels=3)
# pred_image = torch.randn(4, 3, 256, 256)
# target_image = torch.randn(4, 3, 256, 256)
# loss = loss_lap(pred_image, target_image)
# loss.backward()


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss to encourage spatial smoothness in an image.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        """
        Calculates the total variation loss for a given image tensor.
        
        Args:
            img (torch.Tensor): The input image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: The total variation loss.
        """
        # Calculate the absolute differences between a pixel and its horizontal neighbor
        h_variation = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        
        # Calculate the absolute differences between a pixel and its vertical neighbor
        v_variation = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        
        # Sum the differences and return the mean loss
        return torch.mean(h_variation) + torch.mean(v_variation)

class GradientEdgeLoss(nn.Module):
    def __init__(self):
        super(GradientEdgeLoss, self).__init__()
        # Define Sobel kernels for horizontal and vertical gradients
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Store kernels on the specified device (e.g., 'cuda' or 'cpu')
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        
        # Loss function to compare gradient maps (L1 is common)
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        """
        Calculates the gradient loss between prediction and target.
        
        Args:
            pred (torch.Tensor): The predicted image tensor (B, C, H, W).
            target (torch.Tensor): The ground truth image tensor (B, C, H, W).
            
        Returns:
            torch.Tensor: The calculated gradient loss.
        """
        # Assuming input tensors are in grayscale or we operate on the first channel
        # If multichannel, you might want to average gradients across channels
        if pred.shape[1] > 1:
            # A simple way to handle RGB: convert to grayscale
            # Using standard luminosity weights
            pred_gray = 0.299 * pred[:, 0:1, :, :] + 0.587 * pred[:, 1:2, :, :] + 0.114 * pred[:, 2:3, :, :]
            target_gray = 0.299 * target[:, 0:1, :, :] + 0.587 * target[:, 1:2, :, :] + 0.114 * target[:, 2:3, :, :]
        else:
            pred_gray = pred
            target_gray = target

        # Calculate horizontal and vertical gradients for the prediction
        pred_grad_x = F.conv2d(pred_gray, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.kernel_y, padding=1)
        
        # Calculate horizontal and vertical gradients for the target
        target_grad_x = F.conv2d(target_gray, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.kernel_y, padding=1)
        
        # Compute L1 loss between the gradient maps
        loss_x = self.loss(pred_grad_x, target_grad_x)
        loss_y = self.loss(pred_grad_y, target_grad_y)
        
        # Combine the losses from both directions
        return loss_x + loss_y




class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], weight=1.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.selected_layers = layers
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.weight = weight

    def forward(self, pred_img, target_img):
        loss = 0.0
        x = pred_img.repeat(1, 3, 1, 1)  # CT → fake RGB
        y = target_img.repeat(1, 3, 1, 1)
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return self.weight * loss

def fft_loss(pred, target, weight=1.0):
    # pred, target: (B, 1, H, W)
    pred_fft = torch.fft.fft2(pred.squeeze(1))
    target_fft = torch.fft.fft2(target.squeeze(1))
    return weight * torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)))


class MultipleLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, output, target):
        loss = sum(weight*loss(output, target) for loss, weight in zip(self.losses, self.weights))
        return loss

#######
class MultiScaleGradientVarianceLoss(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weights=None, patch_size=4):
        super().__init__()

        # Sobel kernels as conv layers for gradient x and y
        self.grad_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.grad_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        self.grad_x.weight.data = torch.tensor([[[[-1, 0, 1],
                                                  [-2, 0, 2],
                                                  [-1, 0, 1]]]], dtype=torch.float32)
        self.grad_y.weight.data = torch.tensor([[[[-1, -2, -1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]]]], dtype=torch.float32)

        self.grad_x.weight.requires_grad = False
        self.grad_y.weight.requires_grad = False

        self.scales = scales
        self.patch_size = patch_size

        if weights is None:
            self.weights = [1.0 / len(scales)] * len(scales)
        else:
            self.weights = weights

        self.loss = nn.MSELoss(reduction='sum')

    def variance_map(self, img, dim='x'):
        # img shape: (B, 1, H, W)
        grad = self.grad_x(img) if dim == 'x' else self.grad_y(img)
        # Unfold into non-overlapping patches of patch_size x patch_size
        patches = F.unfold(grad, kernel_size=self.patch_size, stride=self.patch_size)
        # patches shape: (B, patch_size*patch_size, num_patches)
        var = patches.var(dim=1, unbiased=False)  # variance over pixels in each patch
        return var  # shape: (B, num_patches)

    def forward(self, pred, target):
        total_loss = 0.0
        for scale, w in zip(self.scales, self.weights):
            if scale != 1:
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target

            loss_x = self.loss(self.variance_map(pred_scaled, 'x'),
                               self.variance_map(target_scaled, 'x'))
            loss_y = self.loss(self.variance_map(pred_scaled, 'y'),
                               self.variance_map(target_scaled, 'y'))

            total_loss += w * (loss_x + loss_y)

        return total_loss

#############
class GradientVarianceLoss(nn.Module):
    def __init__(self, n=4):
        super().__init__()

        self.grad_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.grad_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        # Initialize weights directly
        self.grad_x.weight.data = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.grad_y.weight.data = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)

        self.grad_x.weight.requires_grad = False
        self.grad_y.weight.requires_grad = False
        self.loss = nn.MSELoss(reduction='sum')
        self.n = n

    def forward(self, output, target):
        return self.l2norm(output, target, 'x') + self.l2norm(output, target, 'y')

    def variance_map(self, image, dim='x'):
        B, _, H, W = image.shape

        # Image has the shape (batch x 1 x 362 x 362)
        # Step 1: Calculate Gradient_map (batch x 1 x 362 x 362)
        G = self.grad_x(image) if dim == 'x' else self.grad_y(image)

        # Step 2: Devide G into patches n x n non-overlapping patches
        G= F.unfold(G, kernel_size=self.n, stride=self.n)  # (B, n*n, num_patches)


        var = G.var(dim=1, unbiased=False)

        return var  # (B x w*h/n**2)

    def l2norm(self, output, target, dim='x'):
        var_out = self.variance_map(output, dim=dim)
        var_target = self.variance_map(target, dim=dim)

        return self.loss(var_out, var_target)

#############################
class GaussianEdgeEnhancedLoss(nn.Module):
    def __init__(self, cutoff_freq=0.5, sigma=1, img_size=362):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sigma = sigma

        # Pre-compute the Gaussian high-pass filter for 362x362 images
        self.register_buffer('highpass_filter', self._create_gaussian_highpass_filter(img_size, img_size))

    def _create_gaussian_highpass_filter(self, H, W):
        # Create frequency grids
        u = torch.fft.fftfreq(H).view(-1, 1)
        v = torch.fft.fftfreq(W).view(1, -1)

        # Distance from DC component
        D = torch.sqrt(u ** 2 + v ** 2)

        # Gaussian high-pass filter: 1 - exp(-(D-cutoff_freq)^2 / (2 * sigma^2))
        highpass_filter = 1 - torch.exp(-((D - self.cutoff_freq) ** 2) / (2 * self.sigma ** 2))

        return highpass_filter

    def forward(self, output, target):
        # Remove channel dimension: (batch x 1 x 362 x 362) -> (batch x 362 x 362)
        output_2d = output.squeeze(1)
        target_2d = target.squeeze(1)

        # Compute 2D FFT
        output_fft = torch.fft.fft2(output_2d)
        target_fft = torch.fft.fft2(target_2d)

        # Get magnitude spectra
        output_magnitude = torch.abs(output_fft)
        target_magnitude = torch.abs(target_fft)

        # Apply Gaussian high-pass filter
        output_filtered = output_magnitude * self.highpass_filter
        target_filtered = target_magnitude * self.highpass_filter

        # Compute L1 loss between filtered high-frequency magnitudes
        loss = F.l1_loss(output_filtered, target_filtered)

        return loss
    


# -------------------------
# Second derivative w.r.t x
# -------------------------
def dxx(img):
    """
    Compute second-order derivative along x-axis.
    img: torch.Tensor of shape (B, C, H, W)
    """
    kernel = torch.tensor([[0, 0, 0],
                           [1, -2, 1],
                           [0, 0, 0]], dtype=torch.float32, device=img.device)
    B, C, H, W = img.shape
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=C)

# -------------------------
# Second derivative w.r.t y
# -------------------------
def dyy(img):
    """
    Compute second-order derivative along y-axis.
    img: torch.Tensor of shape (B, C, H, W)
    """
    kernel = torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32, device=img.device)
    B, C, H, W = img.shape
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=C)

# -------------------------
# Mixed derivative dxy
# -------------------------
def dxy(img):
    """
    Mixed derivative ∂²f / ∂x∂y
    """
    # Approximate ∂f/∂y first, then ∂/∂x
    kernel_y = torch.tensor([[1], [0], [-1]], dtype=torch.float32, device=img.device) * 0.5  # 3x1
    kernel_x = torch.tensor([[1, 0, -1]], dtype=torch.float32, device=img.device) * 0.5      # 1x3
    
    _, C, H, W = img.shape
    # First y-derivative
    kernel_y = kernel_y.view(1, 1, 3, 1).repeat(C, 1, 1, 1)
    dy = F.conv2d(img, kernel_y, padding=(1,0), groups=C)
    
    # Then x-derivative
    kernel_x = kernel_x.view(1, 1, 1, 3).repeat(C, 1, 1, 1)
    dxy_res = F.conv2d(dy, kernel_x, padding=(0,1), groups=C)
    return dxy_res

def dyx(img):
    """
    Mixed derivative ∂²f / ∂y∂x
    """
    # Approximate ∂f/∂x first, then ∂/∂y
    kernel_x = torch.tensor([[1, 0, -1]], dtype=torch.float32, device=img.device) * 0.5
    kernel_y = torch.tensor([[1], [0], [-1]], dtype=torch.float32, device=img.device) * 0.5
    
    B, C, H, W = img.shape
    # First x-derivative
    kernel_x = kernel_x.view(1, 1, 1, 3).repeat(C, 1, 1, 1)
    dx = F.conv2d(img, kernel_x, padding=(0,1), groups=C)
    
    # Then y-derivative
    kernel_y = kernel_y.view(1, 1, 3, 1).repeat(C, 1, 1, 1)
    dyx_res = F.conv2d(dx, kernel_y, padding=(1,0), groups=C)
    return dyx_res

# -------------------------
# Mixed derivative dyx
# -------------------------
def dyx(img):
    """
    Compute mixed second-order derivative along y then x.
    img: torch.Tensor of shape (B, C, H, W)
    """
    kernel = torch.tensor([[1, 0, -1],
                           [0, 0, 0],
                           [-1, 0, 1]], dtype=torch.float32, device=img.device) * 0.25
    B, C, H, W = img.shape
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=C)


class SinoLocalStrucLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, output):
        # Compute differences immediately
        Dxx = dxx(target) - dxx(output).mean()
        Dyy = dyy(target) - dyy(output).mean()
        Dxy = dxy(target) - dxy(output).mean()
        Dyx = dyx(target) - dyx(output).mean()

        # Compute local structure loss
        loss = torch.sqrt(Dxx**2 + Dyy**2 + Dxy**2 + Dyx**2)

        # Average over batch, channel, height, width
        return loss.sum()














##############################

class SinoMAP(nn.Module):
    def __init__(self, sigma=10):
        super().__init__()
        self.sigma = sigma
        self.log = torch.Tensor([4096])

    def forward(self,X, X_gt): # The Input Consists of Post_logged and scaled
        X_gt_minuspostlog = CT_library.X_to_minuspostlog(X_gt)
        X_gt_proj = CT_library.minuspostlog_to_proj(X_gt_minuspostlog)

        X_minuspostlog = CT_library.X_to_minuspostlog(X)
        X_proj = CT_library.minuspostlog_to_proj(X_minuspostlog)

        loss = (X_gt_proj - X_proj)**2/(2*self.sigma**2) + X_proj*(-X_gt_minuspostlog - self.log) + torch.log(X_proj)
        return loss.sum()


class WeightedL1L2SinogramLoss(nn.Module):
    def __init__(self, N0=4096, sigma_e=0.0, 
                 lambda_wls=1.0, lambda_l1=0.1, eps=1e-6):
        super().__init__()
        self.N0 = N0
        self.sigma_e = sigma_e
        self.lambda_wls = lambda_wls
        self.lambda_l1 = lambda_l1
        self.eps = eps

    def forward(self, S_pred, S_high):
        """
        S_pred: predicted high-quality sinogram (batch, ...)
        S_high: ground-truth high-quality sinogram (batch, ...)
        """
        # Expected counts from high-dose sinogram
        Lambda = self.N0 * torch.exp(-CT_library.X_to_minuspostlog(S_high))

        # Approximate variance in log-domain
        var = (Lambda + self.sigma_e**2) / (Lambda**2 + self.eps)
        weights = 1.0 / (var + self.eps)

        # Weighted MSE (WLS term)
        wls_loss = 0.5 * torch.mean(weights * (S_pred - S_high)**2)

        # L1 loss
        l1_loss = F.l1_loss(S_pred, S_high)

        # Combined
        total_loss = self.lambda_wls * wls_loss + self.lambda_l1 * l1_loss
        return total_loss




