"""MRI-only inference audit around validated preprocessing and MRIBranch."""
from mcminet.config import default

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import time
import torch
from torchvision.models import ResNet18_Weights

from mcminet.data.mri_preprocessing import preprocess_mri
from mcminet.models.mri_branch import MRIBranch
from mcminet.models.mri_roi_encoder import MRIROIEncoder

ROI_NAMES = ('tumor', 'peritumoral', 'lymph_node')
SMOKE_SEED = 20260910


@dataclass(frozen=True)
class MRISources:
    """Only the four attributes consumed by validated preprocess_mri; no label/WSI."""
    patient_id: str
    mri_dicom_dir: Path
    tumor_mask_path: Path
    lymph_node_mask_path: Path


def load_mri_sources(manifest):
    manifest=Path(manifest).resolve();records=[];seen=set()
    required=tuple(MRISources.__dataclass_fields__)
    with manifest.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if len(reader.fieldnames or [])!=len(set(reader.fieldnames or [])) or not set(required)<=set(reader.fieldnames or []):
            raise ValueError('Missing/duplicate MRI source columns')
        for row in reader:
            values={k:(row.get(k) or '').strip() for k in required}
            if not all(values.values()) or values['patient_id'] in seen:raise ValueError('Invalid/duplicate MRI source record')
            seen.add(values['patient_id'])
            records.append(MRISources(values['patient_id'],**{k:(manifest.parent/values[k]).resolve() for k in required[1:]}))
    if not records:raise ValueError('Empty MRI manifest')
    return tuple(records)


def state_fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        cpu=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((cpu.dtype,tuple(cpu.shape))).encode());h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def load_mri_model():
    weights=ResNet18_Weights[default("MRI.weights")];filename=weights.url.rsplit('/',1)[-1]
    path=Path(torch.hub.get_dir())/'checkpoints'/filename
    if not path.is_file():raise FileNotFoundError('Required local pretrained weights missing; STOP')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(filename.rsplit('-',1)[-1].split('.')[0]):raise ValueError('Corrupt local pretrained weights; STOP')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SMOKE_SEED)
        model=MRIBranch(pretrained=True,trainable=True).cpu().eval()
    return model,dict(pretrained=True,trainable=True,weights=str(weights),weight_sha256=digest,
        seed=SMOKE_SEED,projection_and_fusion='seeded random initialization; untrained',
        conv1='pretrained RGB channel mean',dropout=default("MRI.fusion_dropout"))


def select_device(device='cpu'):
    if device not in ('cpu','mps','auto'):raise ValueError('device must be cpu, mps or auto')
    available=torch.backends.mps.is_available()
    if device=='mps' and not available:raise RuntimeError('MPS unavailable')
    return 'mps' if device=='mps' or (device=='auto' and available) else 'cpu'


def model_inputs(preprocessing):
    """Preserve validated [1,H,W] tensors; add one batch dimension only."""
    rois=[]
    for name in ROI_NAMES:
        value=getattr(preprocessing,name+'_roi')
        if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.shape[0]!=1 or min(value.shape[1:])<=0:
            raise ValueError(name+' validated ROI must have shape [1,H,W], nonempty')
        if value.dtype!=torch.float32:raise TypeError(name+' ROI must be float32')
        if value.device.type!='cpu' or not torch.isfinite(value).all():raise ValueError(name+' ROI must be finite CPU values')
        rois.append(value.unsqueeze(0))
    return tuple(rois)


def preprocess_inputs(sources):
    started=time.perf_counter()
    result=preprocess_mri(sources)  # validated API uses only MRISources attributes
    return result,model_inputs(result),time.perf_counter()-started


def run_mri_smoke(patient_id,rois,model,*,device='cpu'):
    """Call the actual validated fusion twice and audit shared object use, not shapes alone."""
    started=time.perf_counter()
    if not isinstance(patient_id,str) or not patient_id.strip():raise ValueError('patient_id must be nonempty')
    if len(rois)!=3:raise ValueError('Exactly three ordered ROIs required')
    for name,roi in zip(ROI_NAMES,rois):
        if not isinstance(roi,torch.Tensor) or roi.ndim!=4 or roi.shape[:2]!=(1,1) or min(roi.shape[2:])<=0:
            raise ValueError(name+' ROI must be [1,1,H,W], nonempty')
        if roi.dtype!=torch.float32:raise TypeError(name+' ROI must be float32')
        if roi.device.type!='cpu' or not torch.isfinite(roi).all():raise ValueError(name+' ROI must be finite CPU values')
    if sum(isinstance(m,MRIROIEncoder) for m in model.modules())!=1:
        raise ValueError('MRIBranch must contain exactly one shared MRIROIEncoder')
    selected=select_device(device);model.to(selected).eval()
    before=state_fingerprint(model);flags=[p.requires_grad for p in model.parameters()]
    parameter_ids=tuple(id(p) for p in model.roi_encoder.parameters())
    inputs=tuple(roi.to(selected,copy=True) for roi in rois);calls=[];fusions=[]
    def shared_hook(module,args):
        index=len(calls)%3
        if module is not model.roi_encoder or tuple(id(p) for p in module.parameters())!=parameter_ids or not torch.equal(args[0],inputs[index]):
            raise ValueError('Shared encoder identity or ROI order changed')
        if module.training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            raise ValueError('Inference mode not enforced')
        calls.append(ROI_NAMES[index])
    hook=model.roi_encoder.register_forward_pre_hook(shared_hook)
    fusion_hook=model.fusion.register_forward_pre_hook(lambda module,args:fusions.append(args[0].detach().cpu().clone()))
    try:
        inference_start=time.perf_counter()
        with torch.inference_mode():
            first=model(*inputs,return_roi_embeddings=True)
            second=model(*inputs,return_roi_embeddings=True)
        if selected=='mps':torch.mps.synchronize()
        inference_time=time.perf_counter()-inference_start
    finally:
        hook.remove();fusion_hook.remove()
    if calls!=list(ROI_NAMES)*2 or len(fusions)!=2:raise ValueError('Unexpected shared encoder call count')
    for i,output in enumerate((first,second)):
        if tuple(output['roi_embeddings'])!=ROI_NAMES:raise ValueError('ROI output names/order changed')
        expected=torch.cat([output['roi_embeddings'][name].detach().cpu() for name in ROI_NAMES],dim=1)
        if not torch.equal(fusions[i],expected):raise ValueError('Validated ROI concatenation order mismatch')
    outputs={name:first['roi_embeddings'][name].detach().cpu() for name in ROI_NAMES}
    outputs['mri']=first['embedding'].detach().cpu();stats={}
    for name,value in outputs.items():
        other=(second['embedding'] if name=='mri' else second['roi_embeddings'][name]).detach().cpu()
        if value.shape!=(1,default("MRI.patient_embedding_dim")) or not torch.isfinite(value).all() or abs(float(value.norm())-1)>1e-5:
            raise ValueError('Invalid/nonfinite/unnormalized embedding '+name)
        delta=float((value-other).abs().max())
        if not torch.allclose(value,other,atol=1e-6,rtol=1e-6):raise ValueError('MRI repeated inference differs')
        stats[name]=dict(shape=list(value.shape),dtype=str(value.dtype),device='cpu',finite=True,
                         l2_norm=float(value.norm()),repeat_max_absolute_difference=delta)
    if before!=state_fingerprint(model) or flags!=[p.requires_grad for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError('MRI model state/gradient invariant failed')
    return outputs,dict(patient_id=patient_id,device_used=selected,embedding_statistics=stats,
        shared_encoder_instances=1,shared_encoder_call_order=calls,
        identical_parameter_objects_all_calls=True,concatenation_order=list(ROI_NAMES),
        model_parameters_unchanged=True,batchnorm_buffers_unchanged=True,gradients_absent=True,
        dropout_inactive=True,inference_runtime_seconds=inference_time,
        total_audit_runtime_seconds=time.perf_counter()-started,state_sha256=before)
