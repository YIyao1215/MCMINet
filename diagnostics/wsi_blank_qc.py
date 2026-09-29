"""release diagnostics only; no threshold selection or model tensors.

All features use uint8 native 448x448 RGB. Gray is BT.601 luma on encoded
RGB: .299R+.587G+.114B (0..255). HSV uses max/min channel arithmetic,
S=(max-min)/max (zero when max=0), V=max/255, both 0..1. Hue is not needed.
OD=-ln((I+1)/256); low-OD pixels have MEAN channel OD <= threshold.
HSV descriptive score is mean(V*(1-S)); it orders examples, not decisions.
Every hypothetical patch exclusion uses strict fraction > .90.
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from mcminet.data.wsi_preprocessing import NATIVE_SIZE, STRIDE, inspect_metadata, to_rgb

RGB_THRESHOLDS = (180, 190, 200, 210, 220, 230, 240, 245, 250)
V_THRESHOLDS = (.75, .80, .85, .90, .95)
S_THRESHOLDS = (.05, .10, .15, .20, .25)
OD_THRESHOLDS = (.05, .10, .15, .20)


def spatial_sample(dimensions, target=512):
    """Exact min(target,grid size) unique full fields, y-major.

    Evenly spaced rows cover both grid edges. Per-row quotas differ by <=1;
    evenly spaced columns cover both edges when quota>=2. Row count follows
    grid aspect ratio, constrained by available rows/columns. Singletons use
    the grid center. No full-grid allocation, randomness or image selection.
    """
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError('target must be a positive integer')
    if len(dimensions) != 2 or any(int(d) != d or d < 0 for d in dimensions):
        raise ValueError('invalid dimensions')
    nx, ny = (int(d)//STRIDE for d in dimensions)
    n = min(target, nx*ny)
    if not n:
        return []
    rows = min(ny, n, max((n+nx-1)//nx, round(np.sqrt(n*ny/nx))))
    def positions(size, count):
        return [size//2] if count == 1 else np.rint(np.linspace(0, size-1, count)).astype(int).tolist()
    result = []
    for i, y in enumerate(positions(ny, rows)):
        quota = n//rows + (i < n % rows)
        result.extend((x*STRIDE, y*STRIDE) for x in positions(nx, quota))
    return result


def hsv_sv(rgb):
    """Deterministic HSV saturation and value, float64 in [0,1]."""
    scaled = np.asarray(rgb, dtype=np.float64)/255
    maximum, minimum = scaled.max(axis=-1), scaled.min(axis=-1)
    saturation = np.divide(maximum-minimum, maximum, out=np.zeros_like(maximum), where=maximum != 0)
    return saturation, maximum


def patch_features(rgb):
    array = np.asarray(rgb)
    if array.shape != (NATIVE_SIZE, NATIVE_SIZE, 3) or array.dtype != np.uint8:
        raise ValueError('QC requires native 448x448 uint8 RGB, before resizing')
    result = {}
    def stats(prefix, values, quantiles=(10,25,50,75,90)):
        result[prefix+'_mean'] = float(np.mean(values))
        result[prefix+'_median'] = float(np.median(values))
        result.update({prefix+f'_p{q}':float(v) for q,v in zip(quantiles,np.percentile(values,quantiles))})
    for i, channel in enumerate('rgb'):
        stats('rgb_'+channel, array[..., i])
        result['rgb_mean_'+channel] = result['rgb_'+channel+'_mean']
    gray = array.astype(np.float64) @ np.array([.299,.587,.114])
    stats('gray',gray,(10,50,90))
    minimum = array.min(axis=-1)
    for threshold in RGB_THRESHOLDS:
        result[f'rgb_blank_{threshold}'] = float(np.mean(minimum >= threshold))
    s,v = hsv_sv(array)
    stats('hsv_s',s)
    stats('hsv_v',v,(10,50,90))
    result['hsv_descriptive_score'] = float(np.mean(v*(1-s)))
    for vt in V_THRESHOLDS:
        for st in S_THRESHOLDS:
            result[f'hsv_bg_v{round(vt*100):03d}_s{round(st*100):03d}'] = float(np.mean((v>=vt)&(s<=st)))
    od = -np.log((array.astype(np.float64)+1)/256)
    result['od_mean'],result['od_median'] = float(od.mean()),float(np.median(od))
    for threshold in OD_THRESHOLDS:
        result[f'od_low_fraction_{round(threshold*100):03d}'] = float(np.mean(od.mean(axis=-1)<=threshold))
    return result


def manifest_slides(path, patient_ids=('P001','P002')):
    """Project only ID/path columns; never parse, validate or analyze labels."""
    path = Path(path).resolve()
    selected = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'patient_id','wsi_path'} <= set(reader.fieldnames or []):
            raise ValueError('missing QC manifest columns')
        for row in reader:
            pid = row['patient_id'].strip()
            if pid in patient_ids:
                if pid in selected or not row['wsi_path'].strip():
                    raise ValueError('duplicate ID or empty slide path')
                selected[pid] = (path.parent/row['wsi_path'].strip()).resolve()
    if set(selected) != set(patient_ids):
        raise ValueError('requested anonymous IDs missing')
    return [(pid, selected[pid]) for pid in patient_ids]


def collect_qc(slide, patient_id, target=512):
    metadata = inspect_metadata(slide)
    rows, thumbnails = [], []
    for index, (x,y) in enumerate(spatial_sample(metadata.dimensions,target)):
        rgb = to_rgb(slide.read_region((x,y),0,(NATIVE_SIZE,NATIVE_SIZE)))
        features = patch_features(np.asarray(rgb))  # must precede display resize
        rows.append(dict(patient_id=patient_id,sample_index=index,x=x,y=y,**features))
        thumbnails.append(rgb.resize((112,112),Image.Resampling.BILINEAR))
    return rows, thumbnails, metadata


def distribution(values):
    keys = ('min','p10','p25','median','p75','p90','max')
    return dict(zip(keys,map(float,np.percentile(values,[0,10,25,50,75,90,100]))))


def summarize(rows):
    if not rows:
        raise ValueError('no full fields available for QC')
    result = {}
    for key in rows[0]:
        if key in ('patient_id','sample_index','x','y'):
            continue
        values = np.array([row[key] for row in rows])
        result[key] = distribution(values)
        if key.startswith(('rgb_blank_','hsv_bg_','od_low_fraction_')):
            count = int(np.sum(values>.90))
            result[key].update(hypothetical_excluded_count=count,hypothetical_excluded_percent=100*count/len(rows))
    return result


def example_indices(rows):
    n=len(rows)
    order=lambda key: sorted(range(n),key=lambda i:(-rows[i][key],i))
    # Evenly spaced row-order subset of the already spatially stratified sample.
    return dict(spatial=np.rint(np.linspace(0,n-1,min(64,n))).astype(int).tolist(),
        lowest_rgb220=sorted(range(n),key=lambda i:(rows[i]['rgb_blank_220'],i))[:32],
        highest_rgb220=order('rgb_blank_220')[:32],
        hsv_background_like=order('hsv_descriptive_score')[:32],
        high_od=order('od_mean')[:32])


def write_csv(path, rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(path, rows, thumbnails, indices, title):
    width,height=184,150
    sheet=Image.new('RGB',(8*width,40+((len(indices)+7)//8)*height),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((8,10),title,fill='black')
    for slot,index in enumerate(indices):
        x,y=(slot%8)*width,40+(slot//8)*height
        sheet.paste(thumbnails[index],(x,y))
        row=rows[index]
        draw.text((x,y+114),f"{row['patient_id']} QC {row['sample_index']}\n[{row['x']},{row['y']}]",fill='black')
    sheet.save(path)
