"""release exact vector coverage; no patch assignment or model input.

The validated 448/448 full-field grid is reused directly. Shapely closed boxes
represent [x,x+448) x [y,y+448) for area: their boundaries have zero area.
No coordinate rounding, raster approximation, repair or precision grid.
Only excursions outside [0,1] of at most 1e-12 are clamped; small positive
coverage is retained. 'Touching' summary counts mean positive AREA coverage;
separate geometric-contact counts include boundary-only contact.
"""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from mcminet.data.wsi_preprocessing import patch_grid, NATIVE_SIZE, STRIDE, to_rgb
from mcminet.data.qupath_annotation_adapter import bounded_overview, overlay_point

PATCH_AREA = NATIVE_SIZE * NATIVE_SIZE
TOLERANCE = 1e-12
THRESHOLDS = (.10, .25, .50, .70, .80, .90)
EXAMPLE_TARGETS = (.25, .50, .70, .90)
COLUMNS = ('patient_id', 'x', 'y', 'tumor_coverage', 'stroma_coverage',
           'annotated_effective_coverage', 'unannotated_fraction')


@dataclass(frozen=True)
class EffectiveRegions:
    tumor_union: BaseGeometry
    stroma_union: BaseGeometry
    tumor_effective: BaseGeometry
    stroma_effective: BaseGeometry
    annotated_effective: BaseGeometry
    warnings: tuple[str, ...]


def effective_regions(annotation_set, expect_stroma=True):
    """Derived geometry only. Never mutate or replace canonical annotations.

    Invalid, unknown-class or out-of-bounds inputs fail explicitly. Empty Tumor
    fails; complete removal of expected Stroma is retained as an explicit
    warning/NOT READY state with measurable zero Stroma coverage.
    """
    annotations = annotation_set.annotations
    if any(a.classification not in ('Tumor', 'Stroma') for a in annotations):
        raise ValueError('Unexpected/missing annotation classification requires review')
    domain = shapely.box(0, 0, *annotation_set.slide_dimensions)
    for a in annotations:
        g = a.geometry
        if (g is None or not a.is_valid or g.is_empty or not g.is_valid
                or not math.isfinite(g.area) or g.area <= 0
                or not domain.covers(g)):
            raise ValueError('Invalid/empty/out-of-bounds source geometry; no repair or omission')
    tumor = unary_union([a.geometry for a in annotations if a.classification == 'Tumor'])
    stroma = unary_union([a.geometry for a in annotations if a.classification == 'Stroma'])
    if tumor.is_empty:
        raise ValueError('Tumor union is empty; geometry review required')
    if expect_stroma and stroma.is_empty:
        raise ValueError('Expected Stroma union is empty; geometry review required')
    effective_stroma = stroma.difference(tumor)
    annotated = tumor.union(effective_stroma)
    for geometry in (tumor, stroma, effective_stroma, annotated):
        if not geometry.is_valid or not math.isfinite(geometry.area):
            raise ValueError('Invalid/nonfinite derived geometry; no automatic repair')
    overlap = tumor.intersection(effective_stroma).area
    if overlap > PATCH_AREA*TOLERANCE:
        raise ValueError('Derived effective regions have positive-area overlap beyond numerical tolerance')
    warnings = ('Expected effective Stroma is empty after subtraction; manual review required',) if expect_stroma and effective_stroma.is_empty else ()
    return EffectiveRegions(tumor, stroma, tumor, effective_stroma, annotated, warnings)


def _unit_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < -TOLERANCE).any() or (values > 1+TOLERANCE).any():
        raise ValueError('Coverage outside [0,1] beyond numerical tolerance')
    return np.clip(values, 0., 1.)


def measure_coverage(dimensions, regions):
    """Geometry-only, no slide reader. Exact intersections only for contact.

    Vectorized prepared predicates filter disjoint patches, retaining exact
    intersections for every contacting box. All validated grid positions remain
    in the result, including zero-coverage fields. No RGB is requested.
    """
    if len(dimensions) != 2 or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in dimensions):
        raise ValueError('Dimensions must be nonnegative integers')
    coordinates = np.asarray(list(patch_grid(dimensions)), dtype=np.int64).reshape(-1, 2)
    x, y = coordinates[:, 0], coordinates[:, 1]
    patches = shapely.box(x, y, x+NATIVE_SIZE, y+NATIVE_SIZE)
    fractions, contacts = [], {}
    for name, region in [('tumor', regions.tumor_effective), ('stroma', regions.stroma_effective),
                         ('either', regions.annotated_effective)]:
        # Preparation is only on newly derived geometry, never source objects.
        shapely.prepare(region)
        contact = shapely.intersects(region, patches)
        areas = np.zeros(len(patches), dtype=np.float64)
        areas[contact] = shapely.area(shapely.intersection(patches[contact], region))
        fractions.append(_unit_fraction(areas/PATCH_AREA))
        contacts[name] = int(contact.sum())
    tumor, stroma, annotated = fractions
    if (tumor+stroma > 1+TOLERANCE).any() or not np.allclose(tumor+stroma, annotated, rtol=0, atol=TOLERANCE):
        raise ValueError('Effective coverage additivity violated; no renormalization performed')
    values = np.column_stack((tumor, stroma, annotated, _unit_fraction(1-annotated)))
    return coordinates, values, contacts


def summarize_coverage(values):
    tumor, stroma, annotated, unannotated = values.T
    def distribution(v):
        positive = v[v > 0]
        keys = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')
        quantiles = np.percentile(positive, [0,10,25,50,75,90,100]) if len(positive) else [None]*7
        return dict(count=len(positive), **dict(zip(keys, [float(v) if v is not None else None for v in quantiles])))
    return dict(full_grid_count=len(values),
        positive_area_counts=dict(tumor=int((tumor>0).sum()), stroma=int((stroma>0).sum()),
            either=int((annotated>0).sum()), neither=int((annotated==0).sum())),
        thresholds=[dict(threshold=t, tumor_count=int((tumor>=t).sum()),
                         stroma_count=int((stroma>=t).sum())) for t in THRESHOLDS],
        mixed_diagnostics=dict(both_positive=int(((tumor>0)&(stroma>0)).sum()),
            both_ge_010=int(((tumor>=.10)&(stroma>=.10)).sum()),
            both_ge_025=int(((tumor>=.25)&(stroma>=.25)).sum()),
            tumor_ge_050_stroma_positive=int(((tumor>=.50)&(stroma>0)).sum()),
            stroma_ge_050_tumor_positive=int(((stroma>=.50)&(tumor>0)).sum())),
        positive_coverage_distributions=dict(tumor=distribution(tumor), stroma=distribution(stroma)))


def write_coverage_csv(path, patient_id, coordinates, values):
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows((patient_id, int(x), int(y), *map(float, v)) for (x,y),v in zip(coordinates,values))


def representative_examples(values, per_target=3):
    """Deterministic nearest positive coverage, ties by validated grid index.

    Each sheet has distinct indices, selected in declared target order. Mixed
    prototypes (.1,.1), (.5,.5), (.75,.1), (.1,.75) use Euclidean distance among
    BOTH-positive fields. These are display-selection cues, not patch classes.
    Targets may be unmet: actual coverage is always displayed.
    """
    if isinstance(per_target, bool) or not isinstance(per_target, int) or per_target < 1:
        raise ValueError('per_target must be a positive integer')
    groups = {}
    for name, column in [('tumor',0), ('stroma',1)]:
        selected, used = [], set()
        candidates = np.flatnonzero(values[:,column]>0)
        for target in EXAMPLE_TARGETS:
            order = sorted(candidates, key=lambda i:(abs(values[i,column]-target), int(i)))
            chosen = [int(i) for i in order if int(i) not in used][:per_target]
            used.update(chosen)
            selected.extend(dict(index=i, target=f'{name} near {target:.2f}') for i in chosen)
        groups[name] = selected
    selected, used = [], set()
    candidates = np.flatnonzero((values[:,0]>0)&(values[:,1]>0))
    for name, target in [('low-low',(.1,.1)), ('near 50/50',(.5,.5)),
                         ('tumor-dominant mixed',(.75,.1)), ('stroma-dominant mixed',(.1,.75))]:
        order = sorted(candidates, key=lambda i:(float(np.sum((values[i,:2]-target)**2)), int(i)))
        chosen = [int(i) for i in order if int(i) not in used][:per_target]
        used.update(chosen)
        selected.extend(dict(index=i, target=name) for i in chosen)
    groups['mixed'] = selected
    return groups


def qc_sheets(slide, patient_id, coordinates, values, examples, output):
    """Read each selected COMPLETE native field once, display resize only.

    No RGB work is used for coverage. No cropping, normalization or tensors.
    At default selection, <=36 unique native fields are decoded per slide.
    """
    output = Path(output)
    selected = sorted({e['index'] for group in examples.values() for e in group})
    thumbnails = {}
    for i in selected:
        coordinate = tuple(map(int, coordinates[i]))
        rgb = to_rgb(slide.read_region(coordinate, 0, (NATIVE_SIZE, NATIVE_SIZE)))
        if rgb.size != (NATIVE_SIZE, NATIVE_SIZE):
            raise ValueError('QC reader returned incorrect native field size')
        thumbnails[i] = rgb.resize((192,192), Image.Resampling.BILINEAR)
    paths = []
    for name, entries in examples.items():
        sheet = Image.new('RGB', (720, 48+max(1, (len(entries)+2)//3)*260), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8,8),patient_id+' '+name+' coverage examples; diagnostic only',fill='black')
        if not entries:
            draw.text((8,60),'No eligible examples in this diagnostic group',fill='black')
        for slot, entry in enumerate(entries):
            i = entry['index']; x,y = (slot%3)*240,48+(slot//3)*260
            sheet.paste(thumbnails[i],(x,y))
            cx,cy = coordinates[i]; t,s = values[i,:2]
            draw.text((x,y+194),f"{entry['target']}\n{patient_id} [{cx},{cy}]\nT={t:.5f} S={s:.5f}",fill='black')
        path = output/f'{patient_id}_{name}_coverage_examples.png'
        sheet.save(path); paths.append(path.name)
    return len(selected), paths


def coverage_overview(slide, patient_id, dimensions, coordinates, examples, output):
    """Sparse selected-example centers on one bounded whole-slide overview."""
    image = bounded_overview(slide)
    draw = ImageDraw.Draw(image)
    colors = {'tumor':'red','stroma':'green','mixed':'blue'}
    for name, entries in examples.items():
        for entry in entries:
            x,y = coordinates[entry['index']] + NATIVE_SIZE/2
            px,py = overlay_point((x,y),dimensions,image.size)
            draw.ellipse((px-3,py-3,px+3,py+3),outline=colors[name],width=2)
    canvas = Image.new('RGB',(max(720,image.width),image.height+40),'white')
    canvas.paste(image,(0,40))
    ImageDraw.Draw(canvas).text((8,8),patient_id+' selected QC centers: red=T targets, green=S targets, blue=mixed examples',fill='black')
    path = Path(output)/f'{patient_id}_coverage_overview.png'
    canvas.save(path)
    return path.name
