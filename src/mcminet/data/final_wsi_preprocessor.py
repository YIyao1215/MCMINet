"""release validated ROI + native HSV filter, no feature/model computation.

Validated geometry/grid/ROI plan and whole-field tensor conversion are composed.
This module owns ONE background rule: S<=.05 AND V>=.80, reject fraction>.90.
No RGB220 call, calibration grid, patch classes or PatientSample integration.
"""
from mcminet.config import default

from dataclasses import dataclass
import hashlib
from pathlib import Path
import csv
import time
import numpy as np
from PIL import Image, ImageDraw
import torch

from mcminet.data.roi_constrained_wsi_extractor import plan_candidates, tensor_memory_bytes
from mcminet.data.wsi_preprocessing import inspect_metadata, to_rgb, patch_tensor, NATIVE_SIZE, MODEL_SIZE
from mcminet.diagnostics.wsi_blank_qc import hsv_sv
from mcminet.data.qupath_annotation_adapter import bounded_overview, overlay_point

BACKGROUND_S_MAX = default("WSI.preprocessing.background_saturation_threshold")
BACKGROUND_V_MIN = default("WSI.preprocessing.background_value_threshold")
BACKGROUND_FRACTION_MAX = default("WSI.preprocessing.background_fraction_max")


@dataclass(frozen=True)
class FinalWSIResult:
    patient_id: str
    patches: torch.Tensor
    coordinates: torch.Tensor
    tumor_coverage: torch.Tensor
    stroma_coverage: torch.Tensor
    combined_effective_coverage: torch.Tensor
    background_fraction: torch.Tensor
    full_grid_count: int
    roi_candidate_count: int
    background_rejected_count: int
    final_retained_count: int
    slide_dimensions: tuple[int, int]
    mpp_x: float | None
    mpp_y: float | None
    warnings: tuple[str, ...]
    candidate_coordinates: np.ndarray
    candidate_coverage: np.ndarray
    candidate_background_fraction: np.ndarray
    retained_candidate_indices: np.ndarray
    rejected_candidate_indices: np.ndarray
    candidate_thumbnails: tuple[Image.Image, ...]
    estimated_tensor_bytes: int
    estimated_working_bytes: int
    runtime_seconds: float
    finite: bool


def background_pixels(s,v):
    """Inclusive fixed pixel comparisons on [0,1] HSV, no epsilon."""
    s,v=np.asarray(s),np.asarray(v)
    if s.shape != v.shape or not s.size or not np.isfinite(s).all() or not np.isfinite(v).all() or np.any((s<0)|(s>1)|(v<0)|(v>1)):
        raise ValueError('HSV S/V must have matching finite [0,1] values')
    return (s<=BACKGROUND_S_MAX)&(v>=BACKGROUND_V_MIN)


def native_background_fraction(rgb):
    array=np.asarray(rgb)
    if array.shape!=(NATIVE_SIZE,NATIVE_SIZE,3) or array.dtype!=np.uint8:
        raise ValueError('Background filtering requires native 448x448 uint8 RGB')
    s,v=hsv_sv(array)
    return float(np.count_nonzero(background_pixels(s,v))/200704)


def retain_background_fraction(fraction):
    value=np.asarray(fraction,dtype=np.float64)
    if not np.isfinite(value).all() or np.any((value<0)|(value>1)):
        raise ValueError('Background fraction outside [0,1]')
    return value<=BACKGROUND_FRACTION_MAX


def final_memory_bytes(retained_count,candidate_count):
    """Final tensor + small RGB thumbnails + bounded scratch; excludes runtime baseline."""
    return tensor_memory_bytes(retained_count)+candidate_count*112*112*3+64*1024**2


def preprocess_final_wsi(slide,annotations,memory_budget_bytes=1024**3,
                         candidate_check=None,background_check=None,progress=None):
    """Two bounded passes, exact final allocation, no normalized rejected images.

    Pass 1 reads only ROI-qualified full native fields, records HSV fractions,
    SHA-256 of decoded RGB and small display thumbnails. Pass 2 rereads ONLY
    retained fields and verifies their RGB digest before validated tensorization.
    This avoids holding all native fields or duplicating full output tensors.
    Checks run before final allocation; callback mismatch stops processing.
    Caller owns slide lifetime. No raw data or tensors are persisted here.
    """
    start=time.perf_counter()
    metadata=inspect_metadata(slide)
    if metadata.dimensions!=annotations.slide_dimensions:
        raise ValueError('Annotation and slide dimensions differ')
    plan=plan_candidates(annotations)
    if candidate_check is not None:candidate_check(plan)
    n=len(plan.coordinates)
    if n==0:raise ValueError('No ROI-qualified candidates; no final tensor produced')
    if memory_budget_bytes<=0 or final_memory_bytes(0,n)>memory_budget_bytes:
        raise MemoryError('Candidate audit storage exceeds memory budget before RGB reads')
    fractions=np.empty(n,dtype=np.float64);thumbs=[];digests=[]
    for i,coordinate in enumerate(plan.coordinates):
        rgb=to_rgb(slide.read_region(tuple(map(int,coordinate)),0,(NATIVE_SIZE,NATIVE_SIZE)))
        fractions[i]=native_background_fraction(rgb)
        digests.append(hashlib.sha256(np.asarray(rgb).tobytes()).digest())
        thumbs.append(rgb.resize((112,112),Image.Resampling.BILINEAR))
        if progress and ((i+1)%250==0 or i+1==n):progress('background audit',i+1,n)
    keep=np.flatnonzero(retain_background_fraction(fractions))
    rejected=np.flatnonzero(~retain_background_fraction(fractions))
    if background_check is not None:background_check(plan,fractions)
    if not len(keep):raise ValueError('All ROI candidates rejected by locked background rule; no final tensor produced')
    estimated=tensor_memory_bytes(len(keep));working=final_memory_bytes(len(keep),n)
    if working>memory_budget_bytes:
        raise MemoryError('Exact final tensor plus bounded work exceeds memory budget; tensor not allocated')
    patches=torch.empty((len(keep),3,MODEL_SIZE,MODEL_SIZE),dtype=torch.float32,device='cpu')
    destination=patches.numpy()  # shared storage, no output duplicate
    for final_index,candidate_index in enumerate(keep):
        coordinate=tuple(map(int,plan.coordinates[candidate_index]))
        rgb=to_rgb(slide.read_region(coordinate,0,(NATIVE_SIZE,NATIVE_SIZE)))
        if rgb.size!=(NATIVE_SIZE,NATIVE_SIZE) or hashlib.sha256(np.asarray(rgb).tobytes()).digest()!=digests[candidate_index]:
            raise ValueError('Native RGB changed between audit and tensorization; stop')
        tensor=patch_tensor(rgb)
        if not np.isfinite(tensor.numpy()).all():raise ValueError('Nonfinite final tensor')
        destination[final_index]=tensor.numpy()
        if progress and ((final_index+1)%250==0 or final_index+1==len(keep)):
            progress('tensorization',final_index+1,len(keep))
    coverage=torch.from_numpy(plan.coverage[keep].copy())
    return FinalWSIResult(plan.patient_id,patches,torch.from_numpy(plan.coordinates[keep].copy()),
        coverage[:,0],coverage[:,1],coverage[:,2],torch.from_numpy(fractions[keep].copy()),
        plan.full_grid_count,n,len(rejected),len(keep),metadata.dimensions,metadata.mpp_x,metadata.mpp_y,
        plan.warnings,plan.coordinates.copy(),plan.coverage.copy(),fractions,keep,rejected,tuple(thumbs),
        estimated,working,time.perf_counter()-start,True)


def qc_indices(result,count=24):
    """Indices refer to original candidate audit; no biological categories."""
    def spaced(indices):
        return [int(indices[i]) for i in np.rint(np.linspace(0,len(indices)-1,min(count,len(indices)))).astype(int)] if len(indices) else []
    keep=result.retained_candidate_indices
    mixed=keep[(result.candidate_coverage[keep,0]>0)&(result.candidate_coverage[keep,1]>0)]
    return dict(rejected_background_examples=spaced(result.rejected_candidate_indices),
        retained_near_boundary_examples=sorted(map(int,keep),key=lambda i:(-result.candidate_background_fraction[i],i))[:count],
        representative_final_retained=spaced(keep),mixed_retained_examples=spaced(mixed))


def write_final_qc(result,slide,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);paths=[]
    for name,indices in qc_indices(result).items():
        image=Image.new('RGB',(800,40+max(1,(len(indices)+3)//4)*190),'white')
        draw=ImageDraw.Draw(image);draw.text((8,8),result.patient_id+' '+name.replace('_',' ')+'; manual review',fill='black')
        if not indices:draw.text((8,50),'No examples in this group',fill='black')
        for slot,i in enumerate(indices):
            x,y=(slot%4)*200,40+(slot//4)*190
            image.paste(result.candidate_thumbnails[i],(x,y))
            cx,cy=result.candidate_coordinates[i];t,s,c=result.candidate_coverage[i]
            draw.text((x,y+114),f'{result.patient_id} [{cx},{cy}]\nT={t:.4f} S={s:.4f}\ncombined={c:.4f}\nbackground={result.candidate_background_fraction[i]:.5f}',fill='black')
        filename=f'{result.patient_id}_{name}.png';image.save(output/filename);paths.append(filename)
    image=bounded_overview(slide);draw=ImageDraw.Draw(image)
    for coordinate in result.coordinates.numpy():
        x,y=overlay_point(coordinate+NATIVE_SIZE / 2,result.slide_dimensions,image.size)
        draw.point((round(x),round(y)),fill='red')
    canvas=Image.new('RGB',(max(600,image.width),image.height+32),'white');canvas.paste(image,(0,32))
    ImageDraw.Draw(canvas).text((8,8),result.patient_id+' final retained field centers; manual review required',fill='black')
    filename=f'{result.patient_id}_final_retained_locations.png';canvas.save(output/filename);paths.append(filename)
    return paths


def summary(result):
    def stats(values):
        v=np.asarray(values)
        return dict(zip(('min','median','p90','max'),map(float,np.percentile(v,[0,50,90,100])))) if len(v) else dict(min=None,median=None,p90=None,max=None)
    return dict(patient_id=result.patient_id,slide_dimensions=list(result.slide_dimensions),
        full_grid_count=result.full_grid_count,roi_candidate_count=result.roi_candidate_count,
        background_rejected_count=result.background_rejected_count,final_retained_count=result.final_retained_count,
        candidate_retention_percent=100*result.final_retained_count/result.roi_candidate_count,
        full_grid_retention_percent=100*result.final_retained_count/result.full_grid_count,
        tensor_shape=list(result.patches.shape),coordinate_shape=list(result.coordinates.shape),
        dtype=str(result.patches.dtype),coordinate_dtype=str(result.coordinates.dtype),device=str(result.patches.device),
        finite=result.finite,first_five_coordinates=result.coordinates[:5].tolist(),
        coverage={name:stats(value.numpy()) for name,value in [('tumor',result.tumor_coverage),
            ('stroma',result.stroma_coverage),('combined',result.combined_effective_coverage)]},
        retained_background_fraction=stats(result.background_fraction.numpy()),
        rejected_background_fraction=stats(result.candidate_background_fraction[result.rejected_candidate_indices]),
        retained_mixed_count=int(((result.tumor_coverage>0)&(result.stroma_coverage>0)).sum()),
        estimated_tensor_bytes=result.estimated_tensor_bytes,estimated_working_bytes=result.estimated_working_bytes,
        runtime_seconds=result.runtime_seconds,mpp_x=result.mpp_x,mpp_y=result.mpp_y,warnings=list(result.warnings),
        native_rgb_reads=result.roi_candidate_count+result.final_retained_count,
        status='READY FOR HUMAN FINAL WSI PREPROCESSING REVIEW')


def write_audit(result,path):
    retained_map={int(index):i for i,index in enumerate(result.retained_candidate_indices)}
    with Path(path).open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['patient_id','candidate_index','final_index','x','y','tumor_coverage',
            'stroma_coverage','combined_effective_coverage','background_fraction','background_rejected'])
        for i,((x,y),(t,s,c)) in enumerate(zip(result.candidate_coordinates,result.candidate_coverage)):
            writer.writerow([result.patient_id,i,retained_map.get(i,''),int(x),int(y),float(t),float(s),float(c),
                             float(result.candidate_background_fraction[i]),i not in retained_map])
