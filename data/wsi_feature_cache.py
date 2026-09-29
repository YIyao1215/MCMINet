"""Versioned local WSI feature artifacts. No labels, split logic or implicit rebuild."""
from mcminet.config import default

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from mcminet.data.wsi_preprocessing import NATIVE_SIZE,STRIDE,MODEL_SIZE,IMAGENET_MEAN,IMAGENET_STD
from mcminet.data.roi_constrained_wsi_extractor import ROI_THRESHOLD
from mcminet.data.final_wsi_preprocessor import BACKGROUND_S_MAX,BACKGROUND_V_MIN,BACKGROUND_FRACTION_MAX
from mcminet.utils.state_fingerprint import state_fingerprint

SCHEMA_VERSION=2
GRAPH_POLICY=default("graph")
COORDINATE_CONVENTION='level-0 top-left [x,y] pixels'
DEPENDENCIES=('data/final_wsi_preprocessor.py','data/wsi_preprocessing.py',
 'data/qupath_annotation_adapter.py','data/roi_constrained_wsi_extractor.py',
 'diagnostics/wsi_annotation_coverage_qc.py','diagnostics/wsi_blank_qc.py')
ROOT=Path(__file__).resolve().parents[1]


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode())
    h.update(value.numpy().tobytes());return h.hexdigest()


def preprocessing_identity():
    """Validated modules expose constants, but no formal complete fingerprint."""
    policy=dict(version=default("WSI.preprocessing.version"),field_size=NATIVE_SIZE,stride=STRIDE,resize=MODEL_SIZE,roi_coverage_min=ROI_THRESHOLD,
        background=dict(s_max=BACKGROUND_S_MAX,v_min=BACKGROUND_V_MIN,pixel_logic='inclusive AND',
                        fraction_max=BACKGROUND_FRACTION_MAX,reject_comparison='>'),
        imagenet_mean=IMAGENET_MEAN.flatten().tolist(),imagenet_std=IMAGENET_STD.flatten().tolist(),
        effective_geometry='Tumor union; Stroma union minus Tumor union',
        resize_method='whole-field PIL bilinear',coordinate_convention=COORDINATE_CONVENTION,
        source_sha256={name:sha256_file(ROOT/name) for name in DEPENDENCIES})
    return dict(policy=policy,sha256=json_digest(policy))


def expected_provenance(patient_id,wsi_path,geojson_path,encoder,checkpoint_sha256):
    if not isinstance(patient_id,str) or not patient_id.strip():raise ValueError('Nonempty patient_id required')
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Cache generation requires fixed eval WSI encoder')
    return dict(patient_id=patient_id,source=dict(wsi_path=str(Path(wsi_path).resolve()),
        wsi_sha256=sha256_file(wsi_path),geojson_path=str(Path(geojson_path).resolve()),
        geojson_sha256=sha256_file(geojson_path)),preprocessing=preprocessing_identity(),
        encoder=dict(architecture='ResNet18 / WSIPatchEncoder raw 512D',state_sha256=state_fingerprint(encoder),
            checkpoint_sha256=checkpoint_sha256,eval=True,
            source_sha256=sha256_file(ROOT/'models/wsi_patch_encoder.py'),
            cache_source_sha256=sha256_file(ROOT/'data/wsi_feature_cache.py'),
            initialization=default('WSI.weights') if encoder.pretrained else 'random_untrained',
            software=dict(torch=str(torch.__version__),
                          torchvision=__import__('torchvision').__version__,pillow=__import__('PIL').__version__)),
        graph_policy=copy.deepcopy(GRAPH_POLICY),
        graph_source_sha256=sha256_file(ROOT/'data/wsi_graph_builder.py'),
        coordinate_convention=COORDINATE_CONVENTION,feature_dimension=default("WSI.node_feature_dim"))


def validate_tensors(features,coordinates):
    if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=default("WSI.node_feature_dim") or not len(features):
        raise ValueError('Features must be nonempty [N,512]')
    if features.device.type!='cpu' or features.dtype!=torch.float32 or features.requires_grad or not torch.isfinite(features).all():
        raise ValueError('Features must be finite CPU float32 without gradients')
    if not isinstance(coordinates,torch.Tensor) or coordinates.shape!=(len(features),2) or coordinates.device.type!='cpu' or coordinates.dtype!=torch.int64:
        raise ValueError('Coordinates must be matching CPU int64 [N,2]')
    if (coordinates<0).any() or (coordinates%STRIDE!=0).any() or len(torch.unique(coordinates,dim=0))!=len(coordinates):
        raise ValueError('Coordinates must be unique nonnegative level-0 grid origins')


def validate_record(record,expected):
    schemas=[
        (expected,{'patient_id','source','preprocessing','encoder','graph_policy','graph_source_sha256','coordinate_convention','feature_dimension'}),
        (expected.get('source',{}),{'wsi_path','wsi_sha256','geojson_path','geojson_sha256'}),
        (expected.get('encoder',{}),{'architecture','state_sha256','checkpoint_sha256','eval','source_sha256','cache_source_sha256','initialization','software'}),
        (expected.get('encoder',{}).get('software',{}),{'torch','torchvision','pillow'}),
    ]
    if any(not isinstance(v,dict) or set(v)!=keys for v,keys in schemas):
        raise ValueError('Invalid provenance schema; extra labels/splits forbidden')
    if expected['encoder']['eval'] is not True:raise ValueError('Fixed encoder must be eval')
    if not isinstance(record,dict) or set(record)!={'schema_version','patient_id','features','coordinates','metadata'}:
        raise ValueError('Invalid cache schema or forbidden extra fields')
    if record['schema_version']!=SCHEMA_VERSION or record['patient_id']!=expected['patient_id']:
        raise ValueError('Incompatible cache schema/patient: v2 provenance required; rebuild legacy caches from trusted inputs')
    f,c=record['features'],record['coordinates'];validate_tensors(f,c)
    meta=record['metadata']
    if not isinstance(meta,dict) or set(meta)!={'provenance','patch_count','feature_dtype','coordinate_dtype','features_sha256','coordinates_sha256'}:
        raise ValueError('Invalid metadata schema')
    if meta['provenance']!=expected:raise ValueError('Stale/incompatible cache provenance')
    if expected['preprocessing']!=preprocessing_identity() or expected['graph_policy']!=GRAPH_POLICY or expected['coordinate_convention']!=COORDINATE_CONVENTION or expected['feature_dimension']!=default('WSI.node_feature_dim'):
        raise ValueError('Expected policy itself is incompatible')
    if meta['patch_count']!=len(f) or meta['feature_dtype']!=str(f.dtype) or meta['coordinate_dtype']!=str(c.dtype):raise ValueError('Cache shape/dtype metadata mismatch')
    if meta['features_sha256']!=tensor_digest(f) or meta['coordinates_sha256']!=tensor_digest(c):raise ValueError('Cache tensor/order integrity mismatch')
    return record


def make_record(features,coordinates,expected):
    validate_tensors(features,coordinates)
    # copy outside inference mode yields ordinary detached tensors usable by autograd downstream.
    with torch.inference_mode(False):f=features.detach().clone();c=coordinates.detach().clone()
    record=dict(schema_version=SCHEMA_VERSION,patient_id=expected['patient_id'],features=f,coordinates=c,
        metadata=dict(provenance=copy.deepcopy(expected),patch_count=len(f),feature_dtype=str(f.dtype),
            coordinate_dtype=str(c.dtype),features_sha256=tensor_digest(f),coordinates_sha256=tensor_digest(c)))
    return validate_record(record,expected)


def save_cache(path,record,expected):
    """Explicit atomic write; never place caches in raw source directories."""
    validate_record(record,expected);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
    try:
        torch.save(record,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_cache(path,expected):
    """Controlled project-owned dictionary; weights_only avoids arbitrary pickle execution.

    Digests detect accidental edits, not a malicious party that rewrites metadata.
    Expected provenance MUST come from trusted current sources/state, not the cache.
    """
    record=torch.load(path,map_location='cpu',weights_only=True)
    return validate_record(record,expected)
