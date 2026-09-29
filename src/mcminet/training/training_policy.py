"""Explicit release parameter policy; no optimizer or training loop."""


def apply_training_policy(canonical,objective):
    canonical.mri_branch.requires_grad_(True)
    canonical.wsi_branch.gat_encoder.requires_grad_(True)
    canonical.classifier.requires_grad_(True)
    canonical.wsi_branch.patch_encoder.requires_grad_(False).eval()
    objective.requires_grad_(True)
    from mcminet.training.mri_batchnorm_policy import apply_mri_batchnorm_policy
    apply_mri_batchnorm_policy(canonical.mri_branch)


def audit_training_policy(model,objective,patch_encoder):
    """Enumerate every parameter once, reject unknown ownership and wrong flags.

    Audit the cached wrapper, the objective's owned proxies, and the separate
    offline patch encoder. Cached training does not register the offline encoder.
    """
    prefixes={
        'model.mri_branch.roi_encoder.backbone.':'MRI ResNet18',
        'model.mri_branch.roi_encoder.projection.':'MRI ROI projection',
        'model.mri_branch.fusion.':'MRI fusion',
        'model.wsi_branch.gat_encoder.gat1.':'WSI GAT1',
        'model.wsi_branch.gat_encoder.gat2.':'WSI GAT2',
        'model.wsi_branch.gat_encoder.bn1.':'WSI GAT BN',
        'model.wsi_branch.gat_encoder.bn2.':'WSI GAT BN',
        'model.wsi_branch.gat_encoder.projection.':'WSI projection',
        'model.classifier.':'MultimodalClassifier',
        'objective.proxy_metric.hybrid_proxies':'Hybrid proxies',
        'objective.proxy_metric.mri_proxies':'MRI-specific proxies',
        'objective.proxy_metric.wsi_proxies':'WSI-specific proxies',
        'offline_patch_encoder.backbone.':'WSI ResNet18',
    }
    rows=[];seen=set()
    for owner,module in [('model',model),('objective',objective),('offline_patch_encoder',patch_encoder)]:
        for name,param in module.named_parameters(remove_duplicate=False):
            full=owner+'.'+name
            categories=[category for prefix,category in prefixes.items() if (full.startswith(prefix) if prefix.endswith('.') else full==prefix)]
            if len(categories)!=1:raise ValueError('Unknown/ambiguous parameter ownership: '+full)
            if id(param) in seen:raise ValueError('Duplicate parameter ownership: '+full)
            seen.add(id(param));category=categories[0];expected=category!='WSI ResNet18'
            if param.requires_grad!=expected:raise ValueError('Trainability policy mismatch: '+full)
            rows.append(dict(name=full,module=full.rsplit('.',1)[0],shape=list(param.shape),
                parameter_count=param.numel(),requires_grad=param.requires_grad,
                expected_category=category,optimizer_candidate=expected))
    if set(r['expected_category'] for r in rows)!=set(prefixes.values()):raise ValueError('Required parameter category missing')
    if any(m.training for m in patch_encoder.modules()):raise ValueError('Fixed offline WSI encoder must remain eval, including BN')
    return rows
