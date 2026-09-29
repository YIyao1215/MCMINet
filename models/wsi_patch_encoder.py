"""ResNet-18 feature extraction for normalized H&E patches."""
from mcminet.config import default


from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class WSIPatchEncoder(nn.Module):
    """Produce one 512-dimensional feature vector per H&E patch.

    Inputs are floating-point, normalized RGB patches shaped [B, 3, H, W]
    (typically [B, 3, 224, 224]). For ImageNet weights, the data pipeline
    should apply the corresponding ImageNet preprocessing. Patch extraction,
    loading, stain normalization, and augmentation are outside this model.

    ResNet-18 retains its global average pooling and flattening operations,
    with its original classification layer replaced by an identity. Outputs
    have shape [B, 512] and can later serve directly as WSI graph node
    features for WSIGATEncoder(input_dim=512); no graph is built here.

    Args:
        pretrained: Use torchvision's default ImageNet ResNet-18 weights.
            Construction may download weights if they are not cached.
            False initializes the backbone without pretrained weights.
        trainable: Enable gradients for backbone parameters. False disables
            parameter gradients only; train/eval mode still controls batch
            normalization running statistics independently.
    """

    def __init__(self, pretrained: bool = default("WSI.pretrained"), trainable: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights[default("WSI.weights")] if pretrained else None
        self.pretrained = pretrained
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(trainable)

    def forward(self, patches: Tensor) -> Tensor:
        """Map normalized RGB patches [B, 3, H, W] to features [B, 512]."""
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape [B, 3, H, W].")
        if patches.shape[0] == 0 or patches.shape[2] == 0 or patches.shape[3] == 0:
            raise ValueError("patches must have nonempty batch and spatial dimensions.")
        if not patches.is_floating_point():
            raise TypeError("patches must be a floating-point tensor.")
        return self.backbone(patches)
