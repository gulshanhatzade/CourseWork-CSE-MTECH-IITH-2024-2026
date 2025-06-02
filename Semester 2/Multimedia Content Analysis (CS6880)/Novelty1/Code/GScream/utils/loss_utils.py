#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp



import torch
from torch import nn

from kornia import metrics



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_masked(network_output, gt, mask):
    return (torch.abs((network_output - gt)) * mask).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def binary_cross_entropy_loss(network_output, gt):
    loss = F.binary_cross_entropy(network_output, gt)
    return loss

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


# depth loss
def compute_scale_and_shift(prediction, target, mask=None):
    if mask==None:
        mask = torch.ones_like(prediction)
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def ScaleAndShiftLoss(prediction, target, mask=None):
    if mask==None:
        mask = torch.ones_like(prediction).cuda()
    scale, shift = compute_scale_and_shift(prediction, target, mask)
    prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    return mse_loss(prediction_ssi, target, mask) + 0.5 * gradient_loss(prediction_ssi, target, mask)
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def my_ssim(image_pred, image_gt, mask=None, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = ssim_loss(image_pred, image_gt, mask, 5)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def ssim_masked(img1, img2, mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_masked(img1, img2, mask, window, window_size, channel, size_average)


def _ssim_masked(img1, img2, mask, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_map * mask

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




# import torch
from torch import nn

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


def _gradient_x(img):
    """
    Compute gradient in x-direction using a Sobel filter
    Works with both 3D and 4D tensors
    """
    # Check if the input has a batch dimension
    if len(img.shape) == 3:
        # Add batch dimension for processing
        img = img.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False
        
    # Create Sobel kernel for x direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    # Handle multi-channel images by processing each channel
    b, c, h, w = img.shape
    img_padded = F.pad(img, (1, 1, 1, 1), mode='replicate')
    
    # Process each channel with the Sobel filter
    grad_list = []
    for ch in range(c):
        # Extract single channel and add channel dimension for conv2d
        channel = img_padded[:, ch:ch+1, :, :]
        grad = F.conv2d(channel, sobel_x, padding=0)
        grad_list.append(grad)
    
    # Combine all channels
    grad_x = torch.cat(grad_list, dim=1)
    
    # Remove batch dimension if it wasn't in the original
    if needs_squeeze:
        grad_x = grad_x.squeeze(0)
        
    return grad_x


def _gradient_y(img):
    """
    Compute gradient in y-direction using a Sobel filter
    Works with both 3D and 4D tensors
    """
    # Check if the input has a batch dimension
    if len(img.shape) == 3:
        # Add batch dimension for processing
        img = img.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False
        
    # Create Sobel kernel for y direction
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    # Handle multi-channel images by processing each channel
    b, c, h, w = img.shape
    img_padded = F.pad(img, (1, 1, 1, 1), mode='replicate')
    
    # Process each channel with the Sobel filter
    grad_list = []
    for ch in range(c):
        # Extract single channel and add channel dimension for conv2d
        channel = img_padded[:, ch:ch+1, :, :]
        grad = F.conv2d(channel, sobel_y, padding=0)
        grad_list.append(grad)
    
    # Combine all channels
    grad_y = torch.cat(grad_list, dim=1)
    
    # Remove batch dimension if it wasn't in the original
    if needs_squeeze:
        grad_y = grad_y.squeeze(0)
        
    return grad_y


# [docs]
def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)
    """
    if not isinstance(idepth, torch.Tensor):
        raise TypeError(f"Input idepth type is not a torch.Tensor. Got {type(idepth)}")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")

    if not len(idepth.shape) == 4:
        raise ValueError(f"Invalid idepth shape, we expect BxCxHxW. Got: {idepth.shape}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError(f"idepth and image shapes must be the same. Got: {idepth.shape} and {image.shape}")

    if not idepth.device == image.device:
        raise ValueError(f"idepth and image must be in the same device. Got: {idepth.device} and {image.device}")

    if not idepth.dtype == image.dtype:
        raise ValueError(f"idepth and image must be in the same dtype. Got: {idepth.dtype} and {image.dtype}")

    # compute the gradients
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def inverse_depth_smoothness_loss_spatial(idepth: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)
    """
    if not isinstance(idepth, torch.Tensor):
        raise TypeError(f"Input idepth type is not a torch.Tensor. Got {type(idepth)}")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")

    if not len(idepth.shape) == 4:
        raise ValueError(f"Invalid idepth shape, we expect BxCxHxW. Got: {idepth.shape}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError(f"idepth and image shapes must be the same. Got: {idepth.shape} and {image.shape}")

    if not idepth.device == image.device:
        raise ValueError(f"idepth and image must be in the same device. Got: {idepth.device} and {image.device}")

    if not idepth.dtype == image.dtype:
        raise ValueError(f"idepth and image must be in the same dtype. Got: {idepth.dtype} and {image.dtype}")

    # compute the gradients
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = (torch.abs(idepth_dx * weights_x)) * mask.unsqueeze(0)[:,:,:,:-1]
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)* mask.unsqueeze(0)[:,:,:-1,:]

    # print(idepth.shape, idepth.dtype, idepth.min(), idepth.max())
    # print(image.shape, image.dtype, image.min(), image.max())
    # print(smoothness_x.shape, smoothness_x.dtype, smoothness_x.min(), smoothness_x.max())
    # print(smoothness_y.shape, smoothness_y.dtype, smoothness_y.min(), smoothness_y.max())
    # print(mask.shape, mask.dtype, mask.min(), mask.max())
    # exit()

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)




# [docs]
class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = InverseDepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)
    """

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return inverse_depth_smoothness_loss(idepth, image)
    





def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    reduction: str = "mean",
    padding: str = "same",
) -> torch.Tensor:
    r"""Function that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim` for details about SSIM.

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> loss = ssim_loss(input1, input2, 5)
    """
    # compute the ssim map
    ssim_map: torch.Tensor = metrics.ssim(img1, img2, window_size, max_val, eps, padding)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    # print(loss.shape) # torch.Size([1, 3, 567, 1008])
    # exit()
    if mask is not None:
        loss = loss[mask]

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss



class SSIMLoss(nn.Module):
    r"""Create a criterion that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim_loss` for details about SSIM.

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> criterion = SSIMLoss(5)
        >>> loss = criterion(input1, input2)
    """

    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, reduction: str = "mean", padding: str = "same"
    ) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)


class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        self.resize = resize
        # Load VGG16 feature detector
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).cuda()
        # Use these layers for feature extraction
        self.layers = ['3', '8', '15', '22']
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features)[:int(self.layers[0]) + 1]),
            nn.Sequential(*list(vgg.features)[int(self.layers[0]) + 1:int(self.layers[1]) + 1]),
            nn.Sequential(*list(vgg.features)[int(self.layers[1]) + 1:int(self.layers[2]) + 1]),
            nn.Sequential(*list(vgg.features)[int(self.layers[2]) + 1:int(self.layers[3]) + 1])
        ]).cuda()
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def preprocess(self, x):
        # Ensure input is in right format with 4 dimensions (B,C,H,W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # Normalize using ImageNet mean and std
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, x, y, mask=None):
        if self.resize:
            # Resize to 224x224 if needed
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Preprocess inputs
        x = self.preprocess(x)
        y = self.preprocess(y)
        
        # Compute loss at each layer
        loss = 0.0
        x_features = [x]
        y_features = [y]
        
        for i, block in enumerate(self.blocks):
            x_features.append(block(x_features[-1]))
            y_features.append(block(y_features[-1]))
        
        # Calculate loss - weighted sum of L2 distance between features
        weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
        for i in range(len(x_features)):
            if mask is not None:
                if i > 0:  # For features beyond the initial input
                    # Resize mask to match feature map size
                    curr_mask = F.interpolate(mask, size=x_features[i].shape[2:], mode='nearest')
                    # Apply mask to features
                    masked_diff = (x_features[i] - y_features[i]) * curr_mask
                    loss += weights[i] * torch.mean(torch.square(masked_diff))
                else:
                    # For original input
                    masked_diff = (x_features[i] - y_features[i]) * mask
                    loss += weights[i] * torch.mean(torch.square(masked_diff))
            else:
                loss += weights[i] * torch.mean(torch.square(x_features[i] - y_features[i]))
                
        return loss

def perceptual_loss(prediction, target, mask=None):
    """
    Compute perceptual loss between prediction and target images using VGG features
    If mask is provided, only calculate loss on masked regions
    """
    global _perceptual_loss_fn
    if '_perceptual_loss_fn' not in globals():
        _perceptual_loss_fn = PerceptualLoss().cuda()
        _perceptual_loss_fn.eval()
    
    # Ensure inputs are in right format
    if len(prediction.shape) == 3:
        prediction = prediction.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    if mask is not None and len(mask.shape) == 3:
        mask = mask.unsqueeze(0)
    
    with torch.no_grad():
        return _perceptual_loss_fn(prediction, target, mask)

def gradient_domain_consistency_loss(prediction, target, mask=None):
    """
    Compute gradient domain consistency loss to ensure smooth transitions,
    especially at object boundaries
    """
    # Compute gradients using Sobel filters
    grad_x_pred = _gradient_x(prediction)
    grad_y_pred = _gradient_y(prediction)
    grad_x_target = _gradient_x(target)
    grad_y_target = _gradient_y(target)
    
    # Calculate difference in gradient space
    grad_diff_x = torch.abs(grad_x_pred - grad_x_target)
    grad_diff_y = torch.abs(grad_y_pred - grad_y_target)
    
    # Apply mask if provided
    if mask is not None:
        # If mask is single channel but input is multi-channel, expand mask
        if mask.shape[1] == 1 and prediction.shape[1] > 1:
            expanded_mask = mask.expand(-1, prediction.shape[1], -1, -1)
        else:
            expanded_mask = mask
        
        # Apply higher weights at boundary regions (detected by gradient of mask)
        mask_grad_x = torch.abs(_gradient_x(expanded_mask))
        mask_grad_y = torch.abs(_gradient_y(expanded_mask))
        
        # Create a boundary emphasis weight (higher at boundaries)
        boundary_weight = 1.0 + 5.0 * (mask_grad_x + mask_grad_y)
        
        # Apply weighted mask to gradient differences
        grad_diff_x = grad_diff_x * boundary_weight * expanded_mask
        grad_diff_y = grad_diff_y * boundary_weight * expanded_mask
    
    # Sum gradient differences
    loss = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
    return loss

