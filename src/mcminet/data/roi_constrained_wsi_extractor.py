"""release: combined effective ROI >=0.70, no image-based exclusion.

Reuse validated exact grid/coverage, native-40x metadata validation, white-alpha
compositing and whole-field PIL bilinear 448->224 ImageNet normalization.
No biological patch labels, feature encoder, graph or PatientSample creation.
"""
from mcminet.config import default

from dataclasses import dataclass
from pathlib import Path
import csv
import time
import numpy as np
import torch
from PIL import Image, ImageDraw

from mcminet.data.wsi_preprocessing import (inspect_metadata, to_rgb, patch_tensor,
    blank_fraction, NATIVE_SIZE, MODEL_SIZE, STRIDE)
from mcminet.data.qupath_annotation_adapter import bounded_overview, overlay_point
from mcminet.diagnostics.wsi_blank_qc import hsv_sv, distribution
from mcminet.diagnostics.wsi_annotation_coverage_qc import effective_regions, measure_coverage

ROI_THRESHOLD = default("WSI.preprocessing.roi_coverage_threshold")
DIAGNOSTIC_NAMES = ('rgb220_blank_fraction', 'rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b',
                    'saturation_mean', 'brightness_mean', 'od_mean')


@dataclass(frozen=True)
class CandidatePlan:
    patient_id: str
    slide_dimensions: tuple[int, int]
    full_grid_count: int
    coordinates: np.ndarray
    coverage: np.ndarray  # [N,3]: T, S, T+S; float64, no normalization
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ROIExtractionResult:
    patient_id: str
    patches: torch.Tensor
    coordinates: torch.Tensor
    tumor_coverage: torch.Tensor
    stroma_coverage: torch.Tensor
    combined_effective_coverage: torch.Tensor
    diagnostics: np.ndarray  # rows align exactly; DIAGNOSTIC_NAMES defines columns
    thumbnails: tuple[Image.Image, ...]
    estimated_tensor_bytes: int
    estimated_working_bytes: int
    extraction_runtime_seconds: float
    all_tensors_finite: bool


def candidate_mask(coverage):
    """Inclusive >=0.70 on T+S, without a threshold epsilon or per-class OR."""
    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values[:,:2]).all():
        raise ValueError('Expected finite per-field Tumor/Stroma coverage')
    ts = values[:,:2]
    if (ts < -1e-12).any() or (ts > 1+1e-12).any():
        raise ValueError('Invalid coverage range')
    ts = np.clip(ts,0,1)
    combined = ts[:,0]+ts[:,1]
    if (combined > 1+1e-12).any():
        raise ValueError('Combined coverage exceeds one')
    return np.clip(combined,0,1) >= ROI_THRESHOLD


def plan_candidates(annotation_set):
    regions = effective_regions(annotation_set)
    coordinates, values, _ = measure_coverage(annotation_set.slide_dimensions, regions)
    selected = candidate_mask(values)
    coords = coordinates[selected].copy()
    ts = values[selected,:2]
    coverage = np.column_stack((ts, np.clip(ts.sum(axis=1),0,1)))
    coords.setflags(write=False); coverage.setflags(write=False)
    return CandidatePlan(annotation_set.patient_id, annotation_set.slide_dimensions,
        len(coordinates), coords, coverage, regions.warnings)


def compare_5d3d(plan, csv_path):
    """Independent selection from validated CSV's T+S, checked before RGB reads.

    Does not substitute per-class counts or the CSV's union column for the
    explicitly locked T+S selection. Order and every coordinate must match.
    """
    expected = []
    with Path(csv_path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['patient_id'] != plan.patient_id:
                raise ValueError('release research ID mismatch')
            if float(row['tumor_coverage'])+float(row['stroma_coverage']) >= ROI_THRESHOLD:
                expected.append((int(row['x']),int(row['y'])))
    expected = np.asarray(expected,dtype=np.int64).reshape(-1,2)
    if not np.array_equal(expected,plan.coordinates):
        raise ValueError('release candidate coordinate mismatch; stop before extraction')
    return dict(exact_coordinate_and_order_match=True, candidate_count=len(expected))


def tensor_memory_bytes(count):
    if isinstance(count,bool) or not isinstance(count,int) or count < 0:
        raise ValueError('Candidate count must be a nonnegative integer')
    return count*3*MODEL_SIZE*MODEL_SIZE*4


def working_memory_bytes(count, keep_thumbnails=True):
    # One output allocation + display thumbnails + conservative bounded scratch.
    return tensor_memory_bytes(count)+(count*112*112*3 if keep_thumbnails else 0)+64*1024**2


def native_diagnostics(rgb):
    """Native uint8 pixels only, before resize/normalization; NEVER filtering.

    Historical all-channel RGB>=220; HSV S/V in [0,1] reuse release.
    OD=-ln((I+1)/256), averaged across native pixels and RGB channels.
    """
    array = np.asarray(rgb)
    if array.shape != (NATIVE_SIZE,NATIVE_SIZE,3) or array.dtype != np.uint8:
        raise ValueError('Diagnostics require native uint8 RGB')
    s,v = hsv_sv(array)
    od = -np.log((array.astype(np.float64)+1)/256)
    return np.array([blank_fraction(array),*array.mean(axis=(0,1)),s.mean(),v.mean(),od.mean()],dtype=np.float64)


def extract_candidates(slide, plan, memory_budget_bytes=1024**3, keep_thumbnails=True,
                       progress=None):
    """Allocate output once, decode qualified fields sequentially, never stack.

    Caller owns slide lifetime. A preflight budget failure occurs before tensor
    allocation or RGB reads. Empty pool fails explicitly. Only candidate fields
    trigger reads; no overview is requested here. QC images reuse thumbnails.
    """
    start = time.perf_counter()
    metadata = inspect_metadata(slide)
    if metadata.dimensions != plan.slide_dimensions:
        raise ValueError('Candidate plan dimensions do not match slide')
    n = len(plan.coordinates)
    if n == 0:
        raise ValueError('Empty ROI-qualified candidate set; no model input produced')
    if (plan.coordinates.shape != (n,2) or plan.coordinates.dtype != np.int64
            or plan.coverage.shape != (n,3) or not candidate_mask(plan.coverage).all()
            or not np.allclose(plan.coverage[:,2],plan.coverage[:,:2].sum(axis=1),atol=1e-12,rtol=0)):
        raise ValueError('Malformed or unqualified candidate plan')
    c = plan.coordinates
    if (np.any(c < 0) or np.any(c%STRIDE) or np.any(c+NATIVE_SIZE > plan.slide_dimensions)
            or len(set(map(tuple,c))) != n
            or list(map(tuple,c)) != sorted(map(tuple,c), key=lambda xy:(xy[1],xy[0]))):
        raise ValueError('Candidate coordinates violate validated full-field grid')
    required = working_memory_bytes(n,keep_thumbnails)
    if memory_budget_bytes <= 0 or required > memory_budget_bytes:
        raise MemoryError('Candidate tensor plus bounded work exceeds memory budget; no allocation/read performed')
    patches = torch.empty((n,3,MODEL_SIZE,MODEL_SIZE),dtype=torch.float32,device='cpu')
    destination = patches.numpy()  # shared view, not a second tensor allocation
    diagnostics = np.empty((n,len(DIAGNOSTIC_NAMES)),dtype=np.float64)
    thumbnails = []
    for i,coordinate in enumerate(plan.coordinates):
        rgb = to_rgb(slide.read_region(tuple(map(int,coordinate)),0,(NATIVE_SIZE,NATIVE_SIZE)))
        if rgb.size != (NATIVE_SIZE,NATIVE_SIZE):
            raise ValueError('Reader did not return complete native field')
        diagnostics[i] = native_diagnostics(rgb)
        tensor = patch_tensor(rgb)
        array = tensor.numpy()
        if not np.isfinite(array).all() or not np.isfinite(diagnostics[i]).all():
            raise ValueError('Nonfinite output; no complete result produced')
        destination[i] = array
        if keep_thumbnails:
            thumbnails.append(rgb.resize((112,112),Image.Resampling.BILINEAR))
        if progress is not None and ((i+1)%250==0 or i+1==n):
            progress(i+1,n)
    coordinates = torch.from_numpy(plan.coordinates.copy()).long()
    coverage = torch.from_numpy(plan.coverage.copy())
    return ROIExtractionResult(plan.patient_id,patches,coordinates,coverage[:,0],coverage[:,1],
        coverage[:,2],diagnostics,tuple(thumbnails),tensor_memory_bytes(n),required,
        time.perf_counter()-start,True)


def example_indices(result, count=24):
    n = len(result.coordinates)
    def spaced(indices):
        return [int(indices[i]) for i in np.rint(np.linspace(0,len(indices)-1,min(count,len(indices)))).astype(int)] if len(indices) else []
    mixed = np.flatnonzero((result.tumor_coverage.numpy()>0)&(result.stroma_coverage.numpy()>0))
    groups = dict(representative=spaced(np.arange(n)), mixed=spaced(mixed))
    for name,column,descending in [('highest_rgb220',0,True),('lowest_saturation',4,False),
                                    ('highest_brightness',5,True),('lowest_od',6,False)]:
        groups[name] = sorted(range(n),key=lambda i:((-1 if descending else 1)*result.diagnostics[i,column],i))[:count]
    return groups


def write_qc(result, slide, dimensions, output):
    """Display-only sheets; never regenerate RGB or use normalized tensors."""
    output = Path(output)
    groups = example_indices(result)
    metric = {'highest_rgb220':0,'lowest_saturation':4,'highest_brightness':5,'lowest_od':6}
    paths = []
    for name,indices in groups.items():
        image = Image.new('RGB',(800,44+max(1,(len(indices)+3)//4)*194),'white')
        draw = ImageDraw.Draw(image)
        draw.text((8,8),result.patient_id+' '+name.replace('_',' ')+' (diagnostic examples only)',fill='black')
        if not indices: draw.text((8,50),'No eligible examples',fill='black')
        for slot,i in enumerate(indices):
            x,y=(slot%4)*200,44+(slot//4)*194
            image.paste(result.thumbnails[i],(x,y))
            cx,cy=result.coordinates[i].tolist()
            text=f'{result.patient_id} [{cx},{cy}]\nT={result.tumor_coverage[i]:.4f} S={result.stroma_coverage[i]:.4f}\ncombined={result.combined_effective_coverage[i]:.4f}'
            if name in metric:
                column=metric[name];text+=f'\n{DIAGNOSTIC_NAMES[column]}={result.diagnostics[i,column]:.4f}'
            draw.text((x,y+114),text,fill='black')
        filename=f'{result.patient_id}_{name}.png';image.save(output/filename);paths.append(filename)
    overview=bounded_overview(slide)
    draw=ImageDraw.Draw(overview)
    for coordinate in result.coordinates.numpy():
        x,y=overlay_point(coordinate+NATIVE_SIZE / 2,dimensions,overview.size)
        draw.point((round(x),round(y)),fill='red')
    canvas=Image.new('RGB',(max(600,overview.width),overview.height+32),'white')
    canvas.paste(overview,(0,32))
    ImageDraw.Draw(canvas).text((8,8),result.patient_id+' ROI-qualified field centers; combined >=0.70',fill='black')
    filename=f'{result.patient_id}_retained_locations.png';canvas.save(output/filename);paths.append(filename)
    return paths


def result_summary(result, plan):
    return dict(patient_id=result.patient_id,slide_dimensions=list(plan.slide_dimensions),
        full_grid_count=plan.full_grid_count,candidate_count=len(result.coordinates),
        retention_percent=100*len(result.coordinates)/plan.full_grid_count,
        tensor_shape=list(result.patches.shape),coordinate_shape=list(result.coordinates.shape),
        tensor_dtype=str(result.patches.dtype),coordinate_dtype=str(result.coordinates.dtype),
        device=str(result.patches.device),all_tensors_finite=result.all_tensors_finite,
        first_five_coordinates=result.coordinates[:5].tolist(),
        coverage_distributions={name:distribution(values.numpy()) for name,values in
            [('tumor',result.tumor_coverage),('stroma',result.stroma_coverage),('combined',result.combined_effective_coverage)]},
        retained_mixed_count=int(((result.tumor_coverage>0)&(result.stroma_coverage>0)).sum()),
        estimated_tensor_bytes=result.estimated_tensor_bytes,estimated_working_bytes=result.estimated_working_bytes,
        extraction_runtime_seconds=result.extraction_runtime_seconds,
        background_distributions={name:distribution(result.diagnostics[:,i]) for i,name in enumerate(DIAGNOSTIC_NAMES)},
        image_diagnostic_exclusion_count=0,warnings=list(plan.warnings))


def write_metadata(result,path):
    with Path(path).open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['patient_id','patch_index','x','y','tumor_coverage','stroma_coverage',
                         'combined_effective_coverage',*DIAGNOSTIC_NAMES])
        for i,(x,y) in enumerate(result.coordinates.tolist()):
            writer.writerow([result.patient_id,i,x,y,float(result.tumor_coverage[i]),
                float(result.stroma_coverage[i]),float(result.combined_effective_coverage[i]),
                *map(float,result.diagnostics[i])])
