"""release MRI-only policy: fixed running statistics, trainable BN affine."""
from torch import nn

MRI_BN_POLICY = "pretrained_running_stats_fixed_affine_trainable_v1"


def mri_batchnorms(mri_branch):
    """Operate only inside the exact shared MRI backbone, never the WSI GAT."""
    modules = [(name, m) for name, m in mri_branch.roi_encoder.backbone.named_modules()
               if isinstance(m, nn.BatchNorm2d)]
    if len(modules) != 20:
        raise ValueError("Expected the validated ResNet18 inventory of 20 MRI BN modules")
    for name, module in modules:
        if (not module.track_running_stats or module.running_mean is None or module.running_var is None
                or not module.affine or not module.weight.requires_grad or not module.bias.requires_grad):
            raise ValueError("MRI BN policy/affine trainability incompatibility: " + name)
    return modules


def apply_mri_batchnorm_policy(mri_branch):
    """Mode change only: no parameter flag/value, buffer value or module replacement."""
    for _, module in mri_batchnorms(mri_branch):
        module.eval()


def audit_mri_batchnorm_policy(mri_branch):
    rows = []
    for name, module in mri_batchnorms(mri_branch):
        if module.training:
            raise ValueError("MRI running-stat policy violated; call the variable model's train()/eval()")
        rows.append(dict(name="mri_branch.roi_encoder.backbone." + name,
                         training=module.training, affine_weight_trainable=module.weight.requires_grad,
                         affine_bias_trainable=module.bias.requires_grad,
                         track_running_stats=module.track_running_stats))
    return rows
