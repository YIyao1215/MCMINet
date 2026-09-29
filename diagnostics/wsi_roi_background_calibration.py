"""release fixed HSV diagnostic grid on validated ROI candidates only.

Native uint8 RGB 448x448; validated HSV conversion has S,V in [0,1]. No model
resize, normalization, features or permanent rejection. Pixel rule is S<=s
AND V>=v, and hypothetical patch rejection is strictly fraction>0.90.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

from mcminet.data.roi_constrained_wsi_extractor import CandidatePlan, candidate_mask
from mcminet.data.wsi_preprocessing import inspect_metadata, to_rgb, blank_fraction, NATIVE_SIZE, STRIDE
from mcminet.diagnostics.wsi_blank_qc import hsv_sv

S_THRESHOLDS = (.02,.03,.04,.05,.06,.07,.08,.10)
V_THRESHOLDS = (.75,.78,.80,.82,.84,.86,.88)
RULES = tuple((s,v) for s in S_THRESHOLDS for v in V_THRESHOLDS)
VISUAL_RULES = (('A',.03,.80),('B',.05,.80),('C',.05,.82),
                ('D',.07,.80),('E',.07,.82),('F',.10,.80))
HUMAN_COMPARISON = ('A','C','D','F')  # predeclared grid coverage, not data-driven
AUX_NAMES = ('s_mean','s_median','v_mean','v_median','od_mean','rgb220_fraction')
BANDS = ('080_085','085_090','090_095','095_100')


@dataclass(frozen=True)
class CalibrationResult:
    patient_id: str
    coordinates: np.ndarray
    coverage: np.ndarray
    fractions: np.ndarray  # [N,56], fixed S-major then V-major order
    auxiliary: np.ndarray
    thumbnails: tuple[Image.Image, ...]


def load_candidate_pool(path, patient_id, dimensions):
    """Read the validated release aligned CSV, never rescan a full grid.

    Reject contaminated/unqualified rows rather than silently changing the
    validated pool. The smoke script also verifies counts, raw hashes and exact
    coordinate equality against validated release metadata.
    """
    coords, coverage = [], []
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for i,row in enumerate(reader):
            if row['patient_id'] != patient_id or int(row['patch_index']) != i:
                raise ValueError('Validated candidate metadata ID/index mismatch')
            coords.append((int(row['x']),int(row['y'])))
            coverage.append([float(row[k]) for k in ('tumor_coverage','stroma_coverage','combined_effective_coverage')])
    coordinates = np.asarray(coords,dtype=np.int64).reshape(-1,2)
    values = np.asarray(coverage,dtype=np.float64).reshape(-1,3)
    _validate_pool(coordinates,values,dimensions)
    coordinates.setflags(write=False); values.setflags(write=False)
    return CandidatePlan(patient_id,tuple(dimensions),(dimensions[0]//STRIDE)*(dimensions[1]//STRIDE),
                         coordinates,values,())


def _validate_pool(coords,coverage,dimensions):
    n = len(coords)
    if n == 0:
        raise ValueError('Zero ROI-qualified patches; no calibration performed')
    if (coords.shape != (n,2) or coords.dtype != np.int64 or coverage.shape != (n,3)
            or not np.isfinite(coverage).all() or not candidate_mask(coverage).all()
            or not np.allclose(coverage[:,:2].sum(axis=1),coverage[:,2],atol=1e-12,rtol=0)):
        raise ValueError('Validated ROI candidate pool must have combined coverage >=0.70')
    if (np.any(coords < 0) or np.any(coords%STRIDE) or np.any(coords+NATIVE_SIZE > dimensions)
            or len(set(map(tuple,coords))) != n
            or list(map(tuple,coords)) != sorted(map(tuple,coords),key=lambda p:(p[1],p[0]))):
        raise ValueError('Validated candidate coordinate/grid mismatch')


def hsv_grid_fractions(s,v):
    """Exact joint threshold counting, without raster approximation.

    searchsorted(left) finds the first S threshold >= each pixel's S;
    searchsorted(right) counts V thresholds <= each V. Joint integer counts
    are cumulatively summed. This is exactly all 56 inclusive comparisons,
    but avoids 56 full-size Boolean arrays per patch. Edges never adapt.
    """
    s,v = np.asarray(s,dtype=np.float64),np.asarray(v,dtype=np.float64)
    if (s.shape != v.shape or not s.size or not np.isfinite(s).all() or not np.isfinite(v).all()
            or np.any((s<0)|(s>1)) or np.any((v<0)|(v>1))):
        raise ValueError('S and V must be matching finite arrays in [0,1]')
    si = np.searchsorted(S_THRESHOLDS,s,side='left')
    vi = np.searchsorted(V_THRESHOLDS,v,side='right')
    counts = np.bincount((si*8+vi).ravel(),minlength=9*8).reshape(9,8)
    cumulative = counts.cumsum(axis=0)[:,::-1].cumsum(axis=1)[:,::-1]
    return (cumulative[:8,1:]/s.size).reshape(-1)


def native_statistics(rgb):
    array = np.asarray(rgb)
    if array.shape != (NATIVE_SIZE,NATIVE_SIZE,3) or array.dtype != np.uint8:
        raise ValueError('Calibration requires native 448x448 uint8 RGB before resizing')
    s,v = hsv_sv(array)
    fractions = hsv_grid_fractions(s,v)
    od = -np.log((array.astype(np.float64)+1)/256)
    auxiliary = np.array([s.mean(),np.median(s),v.mean(),np.median(v),od.mean(),blank_fraction(array)])
    return fractions,auxiliary


def hypothetical_reject(fractions):
    values = np.asarray(fractions,dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Background fractions must be finite and in [0,1]')
    return values > .90


def calibrate_candidates(slide,plan,progress=None):
    metadata = inspect_metadata(slide)
    if metadata.dimensions != plan.slide_dimensions:
        raise ValueError('Validated candidate dimensions differ from slide')
    _validate_pool(plan.coordinates,plan.coverage,plan.slide_dimensions)
    fractions = np.empty((len(plan.coordinates),len(RULES)),dtype=np.float64)
    auxiliary = np.empty((len(plan.coordinates),len(AUX_NAMES)),dtype=np.float64)
    thumbnails = []
    for i,coordinate in enumerate(plan.coordinates):
        rgb = to_rgb(slide.read_region(tuple(map(int,coordinate)),0,(NATIVE_SIZE,NATIVE_SIZE)))
        fractions[i],auxiliary[i] = native_statistics(rgb)  # must precede display resize
        thumbnails.append(rgb.resize((112,112),Image.Resampling.BILINEAR))
        if progress and ((i+1)%250==0 or i+1==len(plan.coordinates)):
            progress(i+1,len(plan.coordinates))
    return CalibrationResult(plan.patient_id,plan.coordinates.copy(),plan.coverage.copy(),
                             fractions,auxiliary,tuple(thumbnails))


def summarize_rules(fractions):
    if fractions.ndim != 2 or fractions.shape[1] != len(RULES) or not len(fractions):
        raise ValueError('Expected nonempty fixed 56-rule results')
    rejected = hypothetical_reject(fractions)
    summaries = []
    for j,(s,v) in enumerate(RULES):
        q = np.percentile(fractions[:,j],[0,10,25,50,75,90,95,100])
        count = int(rejected[:,j].sum())
        summaries.append(dict(s_threshold=s,v_threshold=v,total_count=len(fractions),
            hypothetical_rejected_count=count,exclusion_percent=100*count/len(fractions),
            hypothetical_retained_count=len(fractions)-count,
            **dict(zip(('min','p10','p25','median','p75','p90','p95','max'),map(float,q)))))
    return summaries


def matrices(summaries):
    return {name:np.array([row[name] for row in summaries]).reshape(8,7)
            for name in ('exclusion_percent','hypothetical_retained_count')}


def boundary_examples(values,count=12):
    """Strict below/above 0.90, nearest first, stable retained-index ties.

    Four review bins use the requested exact inequalities; ==0.90 is separately
    counted and retained, and intentionally is in neither strict-boundary list.
    """
    v = np.asarray(values)
    hypothetical_reject(v)
    below = sorted(np.flatnonzero(v<.90),key=lambda i:(-v[i],int(i)))[:count]
    above = sorted(np.flatnonzero(v>.90),key=lambda i:(v[i],int(i)))[:count]
    masks = ((v>=.80)&(v<.85),(v>=.85)&(v<.90),(v>.90)&(v<=.95),(v>.95)&(v<=1))
    bands = {}
    for name,mask in zip(BANDS,masks):
        indices = np.flatnonzero(mask)
        bands[name] = dict(count=len(indices),indices=[int(indices[i]) for i in
            np.rint(np.linspace(0,len(indices)-1,min(3,len(indices)))).astype(int)] if len(indices) else [])
    return dict(below=list(map(int,below)),above=list(map(int,above)),
                exact_090_count=int((v==.90).sum()),bands=bands)


def _sheet(path,result,entries,title):
    # entries: (patch index, rule index, short descriptive text) or None slot.
    image = Image.new('RGB',(800,44+max(1,(len(entries)+3)//4)*202),'white')
    draw = ImageDraw.Draw(image); draw.text((8,8),result.patient_id+' '+title,fill='black')
    if not entries: draw.text((8,54),'No available examples; no substitute rule chosen',fill='black')
    for slot,entry in enumerate(entries):
        x,y=(slot%4)*200,44+(slot//4)*202
        i,j,description=entry
        if i is None:
            draw.text((x,y+30),description,fill='black');continue
        image.paste(result.thumbnails[i],(x,y))
        cx,cy=result.coordinates[i];t,s,c=result.coverage[i];st,vt=RULES[j]
        text=f'{description}\n{result.patient_id} [{cx},{cy}]\nT={t:.4f} S={s:.4f} C={c:.4f}\nbg={result.fractions[i,j]:.5f}\nS<={st:.2f} V>={vt:.2f}'
        draw.text((x,y+114),text,fill='black')
    image.save(path)


def write_artifacts(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reviews={};paths=[]
    for name,s,v in VISUAL_RULES:
        j=RULES.index((s,v));review=boundary_examples(result.fractions[:,j]);reviews[name]=review
        for side in ('below','above'):
            filename=f'{result.patient_id}_rule_{name}_{side}_090.png'
            _sheet(output/filename,result,[(i,j,side+' 0.90') for i in review[side]],
                   f'rule {name}: {side} boundary; diagnostic only')
            paths.append(filename)
        entries=[]
        for band in BANDS:
            indices=review['bands'][band]['indices']
            entries.extend([(i,j,band) for i in indices])
            entries.extend([(None,j,f'{band}: available {review["bands"][band]["count"]}')]+[(None,j,'')]*(3-len(indices)))
        filename=f'{result.patient_id}_rule_{name}_bands.png'
        _sheet(output/filename,result,entries,f'rule {name}: review bands (not classes)');paths.append(filename)
    # F is predeclared most inclusive visual rule: largest S, lowest selected V.
    j=RULES.index((.10,.80))
    indices=sorted(range(len(result.coordinates)),key=lambda i:(-result.fractions[i,j],i))[:24]
    filename=f'{result.patient_id}_false_positive_risk_rule_F.png'
    _sheet(output/filename,result,[(i,j,'High bg; inspect tissue') for i in indices],
           'False-positive risk: rule F ranking, no tissue ground truth');paths.append(filename)
    entries=[]
    for name,s,v in (VISUAL_RULES[1],VISUAL_RULES[5]):
        j=RULES.index((s,v))
        entries.extend((i,j,f'Rule {name}: just below') for i in reviews[name]['below'])
    filename=f'{result.patient_id}_false_negative_risk_B_F.png'
    _sheet(output/filename,result,entries,'False-negative risk: B/F below 0.90; manual review');paths.append(filename)
    return reviews,paths


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_details(result,path):
    with Path(path).open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['patient_id','patch_index','x','y','tumor_coverage','stroma_coverage',
                         'combined_coverage','s_threshold','v_threshold','background_fraction','hypothetical_reject_gt_090'])
        for i,((x,y),(t,s,c)) in enumerate(zip(result.coordinates,result.coverage)):
            for j,(st,vt) in enumerate(RULES):
                fraction=float(result.fractions[i,j])
                writer.writerow([result.patient_id,i,int(x),int(y),float(t),float(s),float(c),st,vt,fraction,fraction>.90])


def write_matrices(output,prefix,summaries):
    for name,values in matrices(summaries).items():
        with (Path(output)/f'{prefix}_{name}_matrix.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['S_threshold / V_threshold',*V_THRESHOLDS])
            writer.writerows((s,*row) for s,row in zip(S_THRESHOLDS,values.tolist()))
