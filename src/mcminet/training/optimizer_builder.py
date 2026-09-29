"""Centralized AdamW ownership derived from the authoritative validated 6A audit."""
import torch
from mcminet.models.cached_batched_mcminet import CachedBatchedMCMINet
from mcminet.training.training_config import validate_objective, finite_number
from mcminet.training.training_policy import audit_training_policy

GROUPS = {
    "MRI_PRETRAINED_BACKBONE": (("MRI ResNet18",), "mri_backbone_lr"),
    "MRI_NEW_LAYERS": (("MRI ROI projection", "MRI fusion"), "mri_new_lr"),
    "WSI_GAT_AND_PROJECTION": (("WSI GAT1", "WSI GAT2", "WSI GAT BN", "WSI projection"), "wsi_lr"),
    "MULTIMODAL_CLASSIFIER": (("MultimodalClassifier",), "classifier_lr"),
    "PROXY_BANKS": (("Hybrid proxies", "MRI-specific proxies", "WSI-specific proxies"), "proxy_lr"),
}


def parameter_groups(model, objective, offline_encoder, config):
    if not isinstance(model, CachedBatchedMCMINet):
        raise ValueError("Only the validated cached-model path is eligible")
    validate_objective(objective, config)
    rows = audit_training_policy(model, objective, offline_encoder)
    parameters = {"model." + n: p for n, p in model.named_parameters(remove_duplicate=False)}
    parameters.update({"objective." + n: p for n, p in objective.named_parameters(remove_duplicate=False)})
    no_decay = set()
    for prefix, module in (("model", model), ("objective", objective)):
        for module_name, child in module.named_modules():
            for local_name, parameter in child.named_parameters(recurse=False):
                if local_name == "bias" or isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                    no_decay.add(id(parameter))
    groups = []
    assigned = set()
    for name, (categories, lr_field) in GROUPS.items():
        names = [r["name"] for r in rows if r["expected_category"] in categories]
        if not names:
            raise ValueError(f"Empty required optimizer group: {name}")
        ps = [parameters[n] for n in names]
        for p in ps:
            if not p.requires_grad or id(p) in assigned:
                raise ValueError("Validated/duplicate parameter in optimizer plan")
            assigned.add(id(p))
        for policy in ("decay", "no_decay"):
            selected = [(n, p) for n,p in zip(names, ps)
                        if (name == "PROXY_BANKS" or id(p) in no_decay) == (policy == "no_decay")]
            if not selected:
                continue
            decay = (config.proxy_weight_decay if name == "PROXY_BANKS" else
                     config.bias_weight_decay if policy == "no_decay" else config.weight_decay)
            groups.append(dict(name=name + "/" + policy, ownership=name,
                               param_names=[n for n,p in selected], params=[p for n,p in selected],
                               lr=getattr(config, lr_field), weight_decay=decay))
    if assigned != {id(p) for p in parameters.values() if p.requires_grad}:
        raise ValueError("Uncategorized or omitted trainable parameter")
    return groups


def build_optimizer(model, objective, offline_encoder, config):
    groups = parameter_groups(model, objective, offline_encoder, config)
    optimizer = torch.optim.AdamW(
        groups, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
        weight_decay=config.weight_decay, foreach=False, fused=False,
    )
    audit_optimizer(optimizer, model, objective, offline_encoder, config)
    return optimizer


def audit_optimizer(optimizer, model, objective, offline_encoder, config):
    """Check exact identity, name and group order, not just parameter counts."""
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Unexpected optimizer implementation")
    expected = parameter_groups(model, objective, offline_encoder, config)
    if len(optimizer.param_groups) != len(expected):
        raise ValueError("Optimizer group count mismatch")
    seen = set()
    rows = []
    for actual, planned in zip(optimizer.param_groups, expected):
        if actual.get("ownership") != planned["ownership"] or actual.get("name") != planned["name"] or actual.get("param_names") != planned["param_names"]:
            raise ValueError("Optimizer group/name/order mismatch")
        if [id(p) for p in actual["params"]] != [id(p) for p in planned["params"]]:
            raise ValueError("Missing, foreign, duplicate or reordered optimizer parameters")
        if actual["weight_decay"] != planned["weight_decay"] or actual["betas"] != (config.adam_beta1, config.adam_beta2) or actual["eps"] != config.adam_eps:
            raise ValueError("Optimizer/config mismatch")
        finite_number("optimizer LR", actual["lr"])
        for option, expected_option in (("foreach", False), ("fused", False), ("amsgrad", False),
                                       ("maximize", False), ("capturable", False), ("differentiable", False)):
            if actual.get(option) != expected_option:
                raise ValueError("Unexpected AdamW option: " + option)
        for name, p in zip(actual["param_names"], actual["params"]):
            if id(p) in seen or not p.requires_grad:
                raise ValueError("Duplicate/validated optimizer parameter")
            seen.add(id(p))
            rows.append(dict(name=name, group=actual["name"], shape=list(p.shape),
                             parameter_count=p.numel(), requires_grad=p.requires_grad))
    return rows


def initial_group_lr(config, group):
    return getattr(config, GROUPS[group["ownership"]][1])
