"""Single-channel 2D MRI ROI representation encoding."""
from mcminet.config import default


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MRIROIEncoder(nn.Module):
    """Map preprocessed MRI ROI crops [B, 1, H, W] to normalized embeddings.

    ResNet-18 produces 512-D pooled features; a linear projection produces
    [B, embedding_dim], followed by L2 normalization. This generic encoder is
    intended for future shared use across tumor, peritumoral, and LN ROIs.
    It performs no intensity normalization, ROI generation, or ROI fusion.

    Args:
        embedding_dim: Positive output width, default 256.
        pretrained: Use torchvision's default ImageNet weights (may download
            if uncached). The new single-channel conv1 uses the mean of the
            pretrained RGB weights across channels. False uses normal PyTorch
            initialization for the new convolution.
        trainable: Enable parameter gradients for backbone AND projection.
            False freezes parameters; BatchNorm running statistics remain
            controlled separately by train()/eval().
    """

    def __init__(
        self, embedding_dim: int = default("MRI.roi_embedding_dim"), pretrained: bool = default("MRI.pretrained"), trainable: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        weights = ResNet18_Weights[default("MRI.weights")] if pretrained else None
        self.pretrained = pretrained
        self.backbone = resnet18(weights=weights)
        rgb_conv = self.backbone.conv1
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                conv1.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.conv1 = conv1
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(backbone_dim, embedding_dim)
        self.requires_grad_(trainable)

    def forward(self, x: Tensor) -> Tensor:
        """Encode floating [B, 1, H, W] crops; return [B, embedding_dim].

        Batch and spatial dimensions must be nonempty. Reasonable spatial
        sizes need not be square or 224x224. Model and input device/dtype must
        match; no implicit image conversion or device transfer is performed.
        """
        if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("x must have shape [B, 1, H, W] (single-channel 2D ROI).")
        if x.shape[0] == 0 or x.shape[2] == 0 or x.shape[3] == 0:
            raise ValueError("x must have nonempty batch and spatial dimensions.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")
        features = self.backbone(x)  # [B, 512], with classifier bypassed.
        return F.normalize(self.projection(features), p=2, dim=1)
