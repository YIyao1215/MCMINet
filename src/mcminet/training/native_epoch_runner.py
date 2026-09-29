"""Native-size MRI epoch adapter; composes validated 6B semantics without editing it."""
import math
import torch
from mcminet.data.cached_patient_dataset import CachedPatientSample, MRI_FIELDS
from mcminet.models.variable_mri_batching import VariableMRICachedMCMINet
from mcminet.training.epoch_runner import EpochResult, LOSS_NAMES
from mcminet.training.optimizer_builder import audit_optimizer
from mcminet.training.training_config import validate_objective
from mcminet.training.training_policy import audit_training_policy
from mcminet.training.mri_batchnorm_policy import audit_mri_batchnorm_policy


def _native_batch(batch, device):
    required={"patient_ids",*MRI_FIELDS,"wsi_features_list","wsi_coordinates_list","labels"}
    if not isinstance(batch,dict) or set(batch)!=required: raise ValueError("Invalid native MRI batch schema")
    ids=batch["patient_ids"]; labels=batch["labels"]
    if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids): raise ValueError("Missing/duplicate patient IDs")
    if not isinstance(labels,torch.Tensor) or labels.dtype!=torch.long or labels.device.type!="cpu" or labels.ndim!=1 or len(labels)!=len(ids): raise ValueError("Invalid native labels")
    if any(not isinstance(batch[name],list) or len(batch[name])!=len(ids) for name in MRI_FIELDS+ ("wsi_features_list","wsi_coordinates_list")):
        raise ValueError("Native list lengths must match")
    samples=[]
    for i,pid in enumerate(ids):
        samples.append(CachedPatientSample(pid,*[batch[name][i] for name in MRI_FIELDS],batch["wsi_features_list"][i],batch["wsi_coordinates_list"][i],int(labels[i])))
    return samples, labels.to(device)


def _run_native(model,objective,batches,optimizer,offline_encoder,config,device):
    if not isinstance(model,VariableMRICachedMCMINet): raise ValueError("VariableMRICachedMCMINet required")
    validate_objective(objective,config); audit_training_policy(model,objective,offline_encoder); audit_mri_batchnorm_policy(model.mri_branch)
    device=torch.device(device)
    if any(t.device!=device for module in (model,objective) for t in list(module.parameters())+list(module.buffers())): raise ValueError("Move model/objective to explicit device")
    training=optimizer is not None
    if training: audit_optimizer(optimizer,model,objective,offline_encoder,config)
    model.train(training); objective.train(training); audit_mri_batchnorm_policy(model.mri_branch)
    totals={name:0.0 for name in LOSS_NAMES}; patients=0; batch_count=0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in batches:
            samples,labels=_native_batch(batch,device)
            if training: optimizer.zero_grad(set_to_none=True)
            rois=tuple([[getattr(sample,name).to(device) for sample in samples] for name in MRI_FIELDS])
            out=model(*rois,batch["wsi_features_list"],batch["wsi_coordinates_list"],return_embeddings=True)
            losses=objective(out["logits"],out["z_mri"],out["z_wsi"],labels)
            if set(losses)!=set(LOSS_NAMES) or any(v.ndim!=0 or not torch.isfinite(v) for v in losses.values()): raise ValueError("Nonfinite/malformed objective")
            if training:
                losses["total"].backward()
                params=[p for group in optimizer.param_groups for p in group["params"]]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    optimizer.zero_grad(set_to_none=True); raise ValueError("Nonfinite gradient; optimizer step not executed")
                optimizer.step(); audit_mri_batchnorm_policy(model.mri_branch)
            count=len(samples)
            for name in LOSS_NAMES: totals[name]+=float(losses[name].detach())*count
            patients+=count; batch_count+=1
    if not patients: raise ValueError("Empty native epoch")
    return EpochResult("training" if training else "validation",patients,batch_count,{name:value/patients for name,value in totals.items()})


def train_native_one_epoch(model,objective,batches,optimizer,offline_encoder,config,*,device="cpu"):
    if optimizer is None: raise ValueError("Native training requires optimizer")
    return _run_native(model,objective,batches,optimizer,offline_encoder,config,device)


def evaluate_native_one_epoch(model,objective,batches,offline_encoder,config,*,device="cpu"):
    return _run_native(model,objective,batches,None,offline_encoder,config,device)
